"""Runtime profiling, hardware metrics, and comparison reports for video runs."""
import csv
import datetime
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict

import cv2
import numpy as np
import torch

from config import *
from .profiler_reports import compare_reports, render_report

logger = logging.getLogger("deepsortVideo")


def log_exception(context_msg):
    logger.error("❌ EXCEPTION in %s", context_msg)
    logger.error(traceback.format_exc())
    for handler in logger.handlers:
        handler.flush()


def _safe_slug(name, fallback="video"):
    base = os.path.splitext(name)[0]
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._")
    return base or fallback

# =====================================================================
#  PIPELINE PROFILER  (added)
#  ---------------------------------------------------------------
#  Har stage ka exact time (CUDA-synced), per-frame deep logs, GPU
#  utilisation/clock/power sampling, final bottleneck report + JSON/CSV
#  files jo alag-alag GPUs par compare kiye ja sakte hain:
#
#      python deepsortVideo.py --compare  run_T4.json run_L4.json ...
#
#  Flags neeche hain. PROFILE_ENABLED=False kar do to sab no-op ho jata hai.
# =====================================================================
def _nvsmi(query):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        return [x.strip() for x in out.split(",")]
    except Exception:
        return None


def _spec_for(gpu_name):
    for key, spec in GPU_SPECS.items():
        if key in (gpu_name or ""):
            return key, spec
    return None, None


def collect_hw_info():
    """GPU/CPU/library spec snapshot -- report + JSON me jata hai taake
    alag GPUs ke results baad me compare ho saken."""
    info = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "is_colab": ("google.colab" in sys.modules) or COLAB_ROOT.exists(),
        "cpu_count": os.cpu_count(),
        "precision": "FP32 (YOLO half=False, OSNet/blink FP32)",
    }
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    try:
        info["cudnn"] = torch.backends.cudnn.version()
        info["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["ram_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                    break
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": p.name,
            "gpu_vram_gb": round(p.total_memory / 2 ** 30, 2),
            "gpu_sm_count": p.multi_processor_count,
            "gpu_compute_capability": f"{p.major}.{p.minor}",
        })
        q = _nvsmi("driver_version,clocks.max.sm,clocks.max.mem,power.limit,pcie.link.gen.current")
        if q and len(q) >= 5:
            info["gpu_driver"] = q[0]
            info["gpu_max_sm_clock_mhz"] = q[1]
            info["gpu_max_mem_clock_mhz"] = q[2]
            info["gpu_power_limit_w"] = q[3]
            info["gpu_pcie_gen"] = q[4]
    else:
        info["gpu_name"] = "CPU-only"
    return info


class _GpuMonitor(threading.Thread):
    """Background sampler: GPU util / clocks / power / temp / throttle.
    Har sample ko us waqt active stage ke naam se tag karta hai, is se
    pata chalta hai konsa stage sach me GPU use kar raha hai aur konsa
    CPU/IO par ruka hua hai."""

    def __init__(self, prof, interval):
        super().__init__(daemon=True)
        self.prof = prof
        self.interval = interval
        self._stop_evt = threading.Event()
        self.samples = []
        self.backend = None
        self._nv = None
        self._h = None
        self.t0 = time.perf_counter()
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.backend = "pynvml"
        except Exception:
            if shutil.which("nvidia-smi"):
                self.backend = "nvidia-smi"
                self.interval = max(interval, 1.0)

    def _sample(self):
        if self.backend == "pynvml":
            nv, h = self._nv, self._h
            u = nv.nvmlDeviceGetUtilizationRates(h)
            d = {
                "util": float(u.gpu),
                "mem_util": float(u.memory),
                "mem_used_mb": nv.nvmlDeviceGetMemoryInfo(h).used / 2 ** 20,
                "sm_clock": float(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)),
            }
            try:
                d["power_w"] = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                pass
            try:
                d["temp_c"] = float(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
            try:
                fn = getattr(nv, "nvmlDeviceGetCurrentClocksThrottleReasons", None) or \
                     getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons")
                d["throttle"] = int(fn(h))
            except Exception:
                pass
            return d
        q = _nvsmi("utilization.gpu,utilization.memory,memory.used,clocks.sm,power.draw,temperature.gpu")
        if not q:
            raise RuntimeError("nvidia-smi failed")
        f = lambda s: float(s) if s.replace(".", "", 1).isdigit() else None
        d = {"util": f(q[0]) or 0.0, "mem_util": f(q[1]) or 0.0,
             "mem_used_mb": f(q[2]) or 0.0, "sm_clock": f(q[3]) or 0.0}
        if f(q[4]) is not None:
            d["power_w"] = f(q[4])
        if f(q[5]) is not None:
            d["temp_c"] = f(q[5])
        return d

    def run(self):
        if not self.backend:
            return
        while not self._stop_evt.wait(self.interval):
            try:
                d = self._sample()
            except Exception:
                continue
            p = self.prof
            c = p.cur
            d["t"] = time.perf_counter() - self.t0
            d["in_frame"] = bool(c is not None and not c["warm"])
            d["stage"] = p.active or "(unstaged)"
            self.samples.append(d)
            if d["in_frame"]:
                p.stage_gpu_util[d["stage"]].append(d["util"])

    def stop(self):
        self._stop_evt.set()

    def summary(self, max_sm_clock=None):
        s = [x for x in self.samples if x["in_frame"]] or self.samples
        if not s:
            return None

        def stat(key):
            v = [x[key] for x in s if key in x]
            if not v:
                return None
            return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}

        out = {"backend": self.backend, "samples": len(s), "util": stat("util"),
               "mem_used_mb": stat("mem_used_mb"), "sm_clock_mhz": stat("sm_clock"),
               "power_w": stat("power_w"), "temp_c": stat("temp_c")}
        thr = [x.get("throttle", 0) for x in s]
        # 0x4 SW power cap, 0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal, 0x80 HW power brake
        bad = 0x4 | 0x8 | 0x20 | 0x40 | 0x80
        out["throttled_samples"] = int(sum(1 for t in thr if t & bad))
        try:
            if max_sm_clock and out["sm_clock_mhz"]:
                out["sm_clock_pct_of_max"] = 100.0 * out["sm_clock_mhz"]["mean"] / float(max_sm_clock)
        except Exception:
            pass
        return out


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_NULL = _NullCtx()


class _Stage:
    __slots__ = ("p", "name", "t0", "prev")

    def __init__(self, p, name):
        self.p = p
        self.name = name

    def __enter__(self):
        p = self.p
        if p.sync:
            torch.cuda.synchronize()
        self.prev = p.active
        p.active = self.name
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        p = self.p
        if p.sync:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - self.t0) * 1000.0
        p.active = self.prev
        p._add(self.name, dt)
        return False


def _find_module(obj):
    if isinstance(obj, torch.nn.Module):
        return obj
    m = getattr(obj, "model", None)
    if isinstance(m, torch.nn.Module):
        return m
    ex = getattr(obj, "extractor", None)
    m = getattr(ex, "model", None)
    if isinstance(m, torch.nn.Module):
        return m
    return None


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else 0.0


class PipelineProfiler:
    def __init__(self):
        self.enabled = PROFILE_ENABLED
        self.sync = bool(PROFILE_SYNC_CUDA and torch.cuda.is_available())
        self.active = None
        self.startup = {}       # module-level / pre-video one-off costs
        self.model_info = {}
        self.hw = None
        self.monitor = None
        self.in_run = False
        self._reset_run("")

    # ---------------- lifecycle ----------------
    def _reset_run(self, video):
        self.video = video
        self.oneoff = {}
        self.frames = []
        self.call_ms = defaultdict(list)
        self.ult = defaultdict(lambda: defaultdict(list))
        self.notes = defaultdict(list)
        self.stage_gpu_util = defaultdict(list)
        self.cur = None
        self.run_meta = {}
        self._totals = []
        self.loop_t0 = None
        self.loop_wall_ms = None
        self.run_t0 = None

    def stage(self, name):
        if not self.enabled:
            return _NULL
        return _Stage(self, name)

    def record(self, name, ms):
        """Manually record an already-measured duration (one-off ya per-frame)."""
        if self.enabled:
            self._add(name, ms)

    def begin_run(self, video):
        if not self.enabled:
            return
        if self.monitor is not None:
            self.monitor.stop()
        self._reset_run(video)
        self.in_run = True
        self.run_t0 = time.perf_counter()
        if self.hw is None:
            self.hw = collect_hw_info()
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            self.monitor = _GpuMonitor(self, PROFILE_GPU_SAMPLE_INTERVAL)
            self.monitor.start()
        h = self.hw
        logger.info("=" * 100)
        logger.info(f"[PROF] RUN START video={video!r}")
        logger.info(f"[PROF] GPU: {h.get('gpu_name')} | VRAM {h.get('gpu_vram_gb')} GB | SMs {h.get('gpu_sm_count')} | "
                    f"CC {h.get('gpu_compute_capability')} | driver {h.get('gpu_driver')} | "
                    f"max SM clk {h.get('gpu_max_sm_clock_mhz')} MHz | power limit {h.get('gpu_power_limit_w')} W")
        logger.info(f"[PROF] CPU: {h.get('cpu_model')} x{h.get('cpu_count')} | RAM {h.get('ram_gb')} GB | "
                    f"torch {h.get('torch')} (CUDA {h.get('torch_cuda')}) | ultralytics {h.get('ultralytics')} | "
                    f"colab={h.get('is_colab')}")
        logger.info(f"[PROF] cuda-sync timing={'ON' if self.sync else 'OFF'} | warmup frames excluded={PROFILE_WARMUP_FRAMES} | "
                    f"GPU monitor={self.monitor.backend if self.monitor else 'none'}")
        self._calibrate()

    def _calibrate(self):
        """Profiler ka apna overhead per stage-call naapo."""
        try:
            n = 300
            t = time.perf_counter()
            for _ in range(n):
                with _Stage(self, "__calib__"):
                    pass
            self.run_meta["profiler_overhead_us_per_stage"] = (time.perf_counter() - t) / n * 1e6
            self.oneoff.pop("__calib__", None)
            self.startup.pop("__calib__", None)
        except Exception:
            pass

    def set_run_meta(self, **kw):
        self.run_meta.update(kw)

    def start_loop(self):
        self.loop_t0 = time.perf_counter()

    def end_loop(self):
        if self.loop_t0 is not None:
            if self.cur is not None:
                self.end_frame()
            self.loop_wall_ms = (time.perf_counter() - self.loop_t0) * 1000.0

    # ---------------- per-frame ----------------
    def begin_frame(self, idx):
        if not self.enabled:
            return
        if self.cur is not None:
            self.end_frame()
        if self.sync:
            torch.cuda.synchronize()
        self.cur = {"idx": idx, "t0": time.perf_counter(), "stages": defaultdict(float),
                    "calls": defaultdict(int), "metrics": defaultdict(float),
                    "warm": idx <= PROFILE_WARMUP_FRAMES}

    def abort_frame(self):
        self.cur = None

    def metric(self, name, value=1):
        if self.cur is not None:
            self.cur["metrics"][name] += value

    def note(self, name, value):
        if self.cur is not None and not self.cur["warm"]:
            self.notes[name].append(float(value))

    def note_ult(self, tag, speed):
        """Ultralytics ka apna preprocess/inference/postprocess split (ms)."""
        if self.cur is None or self.cur["warm"] or not speed:
            return
        for k in ("preprocess", "inference", "postprocess"):
            self.ult[tag][k].append(float(speed.get(k, 0.0)))

    def _add(self, name, dt):
        c = self.cur
        if c is None:
            d = self.oneoff if self.in_run else self.startup
            e = d.setdefault(name, [0.0, 0])
            e[0] += dt
            e[1] += 1
            return
        c["stages"][name] += dt
        c["calls"][name] += 1
        if not c["warm"]:
            self.call_ms[name].append(dt)

    def end_frame(self):
        c = self.cur
        if c is None:
            return
        if self.sync:
            torch.cuda.synchronize()
        total = (time.perf_counter() - c["t0"]) * 1000.0
        staged = sum(c["stages"].values())
        rec = {"frame": c["idx"], "total_ms": total, "unaccounted_ms": total - staged,
               "warm": c["warm"], "stages": dict(c["stages"]), "calls": dict(c["calls"]),
               "metrics": dict(c["metrics"])}
        self.cur = None
        self.frames.append(rec)
        if not rec["warm"]:
            self._totals.append(total)
        if PROFILE_LOG_EVERY_FRAME:
            self._log_frame(rec)
        n = len(self._totals)
        if not rec["warm"] and n >= 10:
            med = float(np.median(self._totals))
            if total > PROFILE_SLOW_FRAME_FACTOR * med:
                top = max(rec["stages"].items(), key=lambda kv: kv[1]) if rec["stages"] else ("?", 0)
                logger.warning(f"[PROF] SLOW FRAME f={rec['frame']} {total:.0f}ms vs median {med:.0f}ms "
                               f"(x{total / med:.1f}) | culprit={top[0]} {top[1]:.0f}ms | "
                               f"bikes={int(rec['metrics'].get('n_live_bikes', 0))}")
        if PROFILE_SUMMARY_EVERY and n and n % PROFILE_SUMMARY_EVERY == 0 and not rec["warm"]:
            self._log_rolling()

    def _log_frame(self, rec):
        m = rec["metrics"]
        items = sorted(rec["stages"].items(), key=lambda kv: -kv[1])
        parts = [f"{k}={v:.1f}" + (f"x{rec['calls'][k]}" if rec["calls"][k] > 1 else "") for k, v in items]
        fps = 1000.0 / rec["total_ms"] if rec["total_ms"] > 0 else 0.0
        logger.info(
            f"[PROF f={rec['frame']:>5}{' WARMUP' if rec['warm'] else ''}] total={rec['total_ms']:8.1f}ms "
            f"({fps:5.2f} fps) | bikes={int(m.get('n_live_bikes', 0))} bike_det={int(m.get('n_bike_dets', 0))} "
            f"mirror_det={int(m.get('n_mirror_dets', 0))} ind_det={int(m.get('n_indicator_dets', 0))} "
            f"plate_det={int(m.get('n_plate_dets', 0))} ocr_calls={int(m.get('n_ocr_calls', 0))} "
            f"reid_crops={int(m.get('n_reid_crops', 0))} | unaccounted={rec['unaccounted_ms']:.1f}ms")
        logger.info("[PROF stages ms] " + "  ".join(parts))
        for h in logger.handlers:
            h.flush()

    def _log_rolling(self):
        fr = [f for f in self.frames if not f["warm"]]
        tot = defaultdict(float)
        for f in fr:
            for k, v in f["stages"].items():
                tot[k] += v
        loop = sum(f["total_ms"] for f in fr) or 1.0
        top = sorted(tot.items(), key=lambda kv: -kv[1])[:5]
        logger.info(f"[PROF ROLLING @ {len(fr)} frames] mean={loop / len(fr):.1f}ms/frame "
                    f"({1000.0 * len(fr) / loop:.2f} fps) | top: " +
                    " | ".join(f"{k} {v / len(fr):.1f}ms ({100 * v / loop:.0f}%)" for k, v in top))

    # ---------------- models ----------------
    def load_model(self, name, loader, **extra):
        """Model load ka time + params + weights ka VRAM footprint."""
        cuda = torch.cuda.is_available()
        m0 = 0
        if cuda:
            try:
                torch.cuda.synchronize()
                m0 = torch.cuda.memory_allocated()
            except Exception:
                cuda = False
        with self.stage(f"load.{name}"):
            obj = loader()
        info = dict(extra)
        try:
            mod = _find_module(obj)
            if mod is not None:
                info["params_M"] = round(sum(p.numel() for p in mod.parameters()) / 1e6, 2)
            if cuda:
                torch.cuda.synchronize()
                info["vram_MB"] = round((torch.cuda.memory_allocated() - m0) / 2 ** 20, 1)
        except Exception:
            pass
        self.model_info[name] = info
        return obj

    # ---------------- report ----------------
    def build_report(self, extra=None):
        frames = [f for f in self.frames if not f["warm"]]
        warm = [f for f in self.frames if f["warm"]]
        if not frames:
            frames, warm = self.frames, []
        n = len(frames)
        rep = {"video": self.video, "hw": self.hw, "meta": dict(self.run_meta), "extra": extra or {},
               "config": {
                   "imgsz": INFERENCE_IMAGE_SIZE, "yolo_model": str(YOLO_BIKE_MODEL_PATH), "mirror_model": str(MIRROR_MODEL_PATH),
                   "indicator_model": str(INDICATOR_MODEL_PATH), "orientation_model": str(ORIENTATION_MODEL_PATH),
                   "osnet_input": list(OSNET_INPUT_SIZE), "min_conf": MIN_CONFIDENCE, "plate_conf": PLATE_CONF,
                   "cuda_sync": self.sync, "warmup_frames": PROFILE_WARMUP_FRAMES},
               "startup_ms": {k: {"ms": v[0], "calls": v[1]} for k, v in self.startup.items()},
               "oneoff_ms": {k: {"ms": v[0], "calls": v[1]} for k, v in self.oneoff.items()},
               "model_info": self.model_info, "frames_counted": n, "warmup_frames_excluded": len(warm)}
        if n == 0:
            return rep

        totals = np.array([f["total_ms"] for f in frames])
        loop_ms = float(totals.sum())
        mean_ms = float(totals.mean())
        rep["frame_time_ms"] = {"mean": mean_ms, "p50": _pct(totals, 50), "p95": _pct(totals, 95),
                                "p99": _pct(totals, 99), "min": float(totals.min()), "max": float(totals.max()),
                                "fps_mean": 1000.0 / mean_ms, "loop_total_s": loop_ms / 1000.0,
                                "loop_wall_s": (self.loop_wall_ms or loop_ms) / 1000.0,
                                "warmup_frame_ms": [float(f["total_ms"]) for f in warm]}
        vfps = float(self.run_meta.get("fps") or 30.0)
        rep["realtime"] = {"video_fps": vfps, "processing_fps": 1000.0 / mean_ms,
                           "realtime_factor": mean_ms / (1000.0 / vfps)}

        names = sorted({k for f in frames for k in f["stages"]})
        stages = {}
        for s in names:
            pf = np.array([f["stages"].get(s, 0.0) for f in frames])
            calls = int(sum(f["calls"].get(s, 0) for f in frames))
            cm = np.array(self.call_ms.get(s) or [0.0])
            gu = self.stage_gpu_util.get(s)
            stages[s] = {
                "calls": calls, "calls_per_frame": calls / n, "total_ms": float(pf.sum()),
                "pct_loop": 100.0 * float(pf.sum()) / loop_ms, "mean_ms_frame": float(pf.mean()),
                "p95_ms_frame": _pct(pf, 95), "mean_ms_call": float(pf.sum()) / max(calls, 1),
                "p50_ms_call": _pct(cm, 50), "p95_ms_call": _pct(cm, 95), "max_ms_call": float(cm.max()),
                "gpu_util_avg": float(np.mean(gu)) if gu else None, "gpu_util_samples": len(gu) if gu else 0}
        unacc = float(sum(f["unaccounted_ms"] for f in frames))
        rep["stages"] = stages
        rep["unaccounted"] = {"total_ms": unacc, "pct_loop": 100.0 * unacc / loop_ms, "mean_ms_frame": unacc / n}

        groups = {}
        for s, v in stages.items():
            g = s.split(".")[0]
            e = groups.setdefault(g, {"total_ms": 0.0, "_gu": []})
            e["total_ms"] += v["total_ms"]
        for s, lst in self.stage_gpu_util.items():
            g = s.split(".")[0]
            if g in groups:
                groups[g]["_gu"].extend(lst)
        for g, e in groups.items():
            e["pct_loop"] = 100.0 * e["total_ms"] / loop_ms
            e["mean_ms_frame"] = e["total_ms"] / n
            e["gpu_util_avg"] = float(np.mean(e["_gu"])) if e["_gu"] else None
            del e["_gu"]
        rep["groups"] = groups

        # model table (Ultralytics ka internal split vs poori call)
        mt = {}
        for tag, d in self.ult.items():
            pre, inf, post = (float(np.mean(d[k])) if d[k] else 0.0 for k in ("preprocess", "inference", "postprocess"))
            call = stages.get(f"{tag}.model_call", {}).get("mean_ms_call", 0.0)
            parse = stages.get(f"{tag}.parse", {}).get("mean_ms_call", 0.0)
            mt[tag] = {"preprocess_ms": pre, "inference_ms": inf, "postprocess_ms": post,
                       "model_call_ms": call, "python_overhead_ms": max(0.0, call - pre - inf - post),
                       "parse_ms": parse}
        rep["yolo_models"] = mt
        rep["notes"] = {k: {"mean": float(np.mean(v)), "max": float(np.max(v)), "n": len(v)}
                        for k, v in self.notes.items() if v}

        # bikes count vs frame time
        nb = np.array([f["metrics"].get("n_live_bikes", 0) for f in frames])
        buckets = []
        for lab, mask in (("0", nb == 0), ("1", nb == 1), ("2", nb == 2), ("3", nb == 3), ("4+", nb >= 4)):
            if mask.any():
                buckets.append({"bikes": lab, "frames": int(mask.sum()), "mean_ms": float(totals[mask].mean())})
        slope = None
        if len(set(nb.tolist())) >= 2:
            try:
                slope = float(np.polyfit(nb, totals, 1)[0])
            except Exception:
                pass
        rep["bike_scaling"] = {"buckets": buckets, "ms_per_extra_bike": slope}

        # slowest frames + histogram
        order = np.argsort(-totals)[:5]
        rep["slowest_frames"] = []
        for i in order:
            f = frames[int(i)]
            top = max(f["stages"].items(), key=lambda kv: kv[1]) if f["stages"] else ("?", 0.0)
            rep["slowest_frames"].append({"frame": f["frame"], "total_ms": f["total_ms"], "culprit": top[0],
                                          "culprit_ms": top[1], "bikes": int(f["metrics"].get("n_live_bikes", 0))})
        hist, edges = np.histogram(totals, bins=8)
        rep["histogram"] = {"counts": hist.tolist(), "edges_ms": [float(e) for e in edges]}

        # GPU monitor + VRAM
        if self.monitor is not None:
            rep["gpu_monitor"] = self.monitor.summary(self.hw.get("gpu_max_sm_clock_mhz") if self.hw else None)
        if torch.cuda.is_available():
            rep["vram"] = {"peak_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
                           "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2 ** 20,
                           "total_mb": (self.hw or {}).get("gpu_vram_gb", 0) * 1024}

        # projections
        rep["projection"] = {lab: {"frames": int(mins * 60 * vfps), "est_minutes": mins * 60 * vfps * mean_ms / 60000.0}
                             for lab, mins in (("1 min video", 1), ("10 min video", 10), ("1 hour video", 60))}
        rep["whatif_gpu"] = self._whatif(stages, n)

        # verdict
        v = []
        top = sorted(stages.items(), key=lambda kv: -kv[1]["total_ms"])[:3]
        for s, e in top:
            f = e["pct_loop"] / 100.0
            speed = 1.0 / (1.0 - f / 2.0) if f < 2 else 1.0
            v.append(f"BOTTLENECK: {s} = {e['pct_loop']:.1f}% of loop ({e['mean_ms_frame']:.1f} ms/frame). "
                     f"Isay 2x fast karne se poori pipeline ~{speed:.2f}x fast hogi.")
        gm = rep.get("gpu_monitor")
        if gm and gm.get("util"):
            u = gm["util"]["mean"]
            if u < 50:
                v.append(f"GPU avg util sirf {u:.0f}% -> pipeline CPU/IO/sync-bound lag rahi hai (GPU idle rehta hai). "
                         f"Bigger GPU lene se zyada faida nahi hoga; pehle CPU-side/IO stages dekho.")
            elif u < 80:
                v.append(f"GPU avg util {u:.0f}% -> mixed: GPU kaafi use ho raha hai par gaps bhi hain (CPU preprocessing / disk / Python).")
            else:
                v.append(f"GPU avg util {u:.0f}% -> GPU-bound. Bari/tez GPU, FP16 (half=True) ya chota imgsz se seedha faida milega.")
            if gm.get("throttled_samples"):
                v.append(f"WARNING: {gm['throttled_samples']} GPU samples me power/thermal throttling mili -> timings pure speed se slow ho sakti hain.")
        if rep["unaccounted"]["pct_loop"] > 8:
            v.append(f"{rep['unaccounted']['pct_loop']:.1f}% time kisi stage me nahi aaya (unaccounted) -> aur instrumentation ya print/log I/O ka overhead ho sakta hai.")
        rep["verdict"] = v
        return rep

    def _whatif(self, stages, n):
        try:
            key, spec = _spec_for((self.hw or {}).get("gpu_name"))
            if not spec:
                return None
            gpu_ms = sum(e["mean_ms_frame"] for s, e in stages.items() if any(h in s for h in GPU_STAGE_HINTS))
            tot = sum(e["mean_ms_frame"] for e in stages.values())
            cpu_ms = max(0.0, tot - gpu_ms)
            out = {"measured_on": key, "gpu_bound_ms_frame": gpu_ms, "cpu_bound_ms_frame": cpu_ms, "targets": {}}
            for k, sp in GPU_SPECS.items():
                ratio = math.sqrt((sp[0] / spec[0]) * (sp[1] / spec[1]))   # geometric mean(TFLOPS ratio, BW ratio)
                est = cpu_ms + gpu_ms / ratio
                out["targets"][k] = {"speed_ratio_vs_measured": ratio, "est_ms_frame": est, "est_fps": 1000.0 / est}
            return out
        except Exception:
            return None

    def save(self, rep, text):
        d = os.path.join(LOG_DIR, PROFILE_SAVE_DIR_NAME)
        os.makedirs(d, exist_ok=True)
        gpu = re.sub(r"[^A-Za-z0-9]+", "-", (self.hw or {}).get("gpu_name", "cpu")).strip("-")
        base = f"{_safe_slug(self.video)}__{gpu}__{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        paths = {"json": os.path.join(d, base + ".json"), "txt": os.path.join(d, base + ".txt"),
                 "frames_csv": os.path.join(d, base + "_frames.csv"), "gpu_csv": os.path.join(d, base + "_gpu_timeline.csv")}
        with open(paths["json"], "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
        with open(paths["txt"], "w", encoding="utf-8") as f:
            f.write(text)
        snames = sorted({k for fr in self.frames for k in fr["stages"]})
        mnames = sorted({k for fr in self.frames for k in fr["metrics"]})
        with open(paths["frames_csv"], "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "warmup", "total_ms", "unaccounted_ms"] + mnames + snames)
            for fr in self.frames:
                w.writerow([fr["frame"], int(fr["warm"]), round(fr["total_ms"], 3), round(fr["unaccounted_ms"], 3)] +
                           [fr["metrics"].get(k, 0) for k in mnames] + [round(fr["stages"].get(k, 0.0), 3) for k in snames])
        if self.monitor is not None and self.monitor.samples:
            keys = ["t", "util", "mem_util", "mem_used_mb", "sm_clock", "power_w", "temp_c", "throttle", "in_frame", "stage"]
            with open(paths["gpu_csv"], "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(keys)
                for s in self.monitor.samples:
                    w.writerow([s.get(k, "") for k in keys])
        else:
            paths.pop("gpu_csv")
        return paths

    def end_run(self, extra=None):
        if not self.enabled:
            return None
        try:
            self.end_loop()
            if self.monitor is not None:
                self.monitor.stop()
                time.sleep(min(0.3, self.monitor.interval))
            if self.run_t0 is not None:
                self.run_meta["run_total_s"] = time.perf_counter() - self.run_t0
            rep = self.build_report(extra)
            text = render_report(rep)
            for line in text.splitlines():
                logger.info(line)
            paths = self.save(rep, text)
            logger.info("[PROF] saved: " + " | ".join(f"{k}={v}" for k, v in paths.items()))
            logger.info("[PROF] Dusri GPU par same video chalao, phir:  python deepsortVideo.py --compare "
                        f"{paths['json']} <dusra_run.json>")
            self.in_run = False
            for h in logger.handlers:
                h.flush()
            return rep
        except Exception:
            log_exception("profiler end_run / report")
            self.in_run = False
            return None


P = PipelineProfiler()


