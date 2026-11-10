import time
import subprocess
import traceback
import logging
import gc
import re
import csv
import shutil
from pathlib import Path
from collections import deque, defaultdict
import os
import cv2
import numpy as np
from ultralytics import YOLO
from config import *

# ------------------------------------------------------------------
# LOGGING SETUP — writes to BOTH the console and a log file, so if the
# Colab process/session dies silently we still have a file on disk
# (flushed after every line) showing exactly the last thing that
# happened before it died. Look at this file first when debugging a
# silent stop; its location is configured in config.py.
# ------------------------------------------------------------------
_LOG_DIR_FOR_LOGGING = str(LOG_DIR)
os.makedirs(_LOG_DIR_FOR_LOGGING, exist_ok=True)
_log_file_path = os.path.join(_LOG_DIR_FOR_LOGGING, "run_debug.log")

logger = logging.getLogger("deepsortVideo")
logger.setLevel(getattr(logging, LOG_LEVEL))
logger.handlers.clear()

_file_handler = logging.FileHandler(_log_file_path, mode="a", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_file_handler)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_console_handler)


def log_exception(context_msg):
    """Call this inside an `except Exception as e:` block. Prints the
    context message + the FULL traceback (not just str(e)) to both
    console and the log file, and flushes immediately so nothing is
    lost even if the process is killed right after."""
    logger.error(f"❌ EXCEPTION in {context_msg}")
    logger.error(traceback.format_exc())
    for h in logger.handlers:
        h.flush()
import os;
from torchvision import transforms
from blink_model import build_model, load_weights,ModelConfig
# --- Deep SORT (nwojke/deep_sort) imports ---
# Make sure `deep_sort` repo folder is in PYTHONPATH or installed as package
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort import preprocessing  # NMS on detections before tracker.update
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "deep-person-reid"))
from torchreid.utils import FeatureExtractor  # pip install torchreid
import torch
from plate_detector import (
    detect_plates_in_bike_crop,
    read_plate_text_from_image,
    warm_up_ocr,
    warm_up_plate_detector,
)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
device = ("cuda" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else DEVICE
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'weights_only': False})

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
import threading
import json
import platform
import datetime
import glob
import math

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


def _bar(frac, width=22):
    n = max(0, min(width, int(round(frac * width))))
    return "#" * n + "." * (width - n)


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


def render_report(rep):
    L = []
    W = 118
    hr = "=" * W

    def sec(t):
        L.append("")
        L.append(hr)
        L.append(f" {t}")
        L.append(hr)

    h = rep.get("hw") or {}
    m = rep.get("meta") or {}
    sec(f"PROFILE REPORT  |  video={rep.get('video')}  |  GPU={h.get('gpu_name')}")
    L.append(f" GPU   : {h.get('gpu_name')} | VRAM {h.get('gpu_vram_gb')} GB | {h.get('gpu_sm_count')} SMs | CC {h.get('gpu_compute_capability')} "
             f"| driver {h.get('gpu_driver')} | max SM clk {h.get('gpu_max_sm_clock_mhz')} MHz | power limit {h.get('gpu_power_limit_w')} W | PCIe gen {h.get('gpu_pcie_gen')}")
    L.append(f" CPU   : {h.get('cpu_model')} x{h.get('cpu_count')} | RAM {h.get('ram_gb')} GB | Colab={h.get('is_colab')}")
    L.append(f" Libs  : torch {h.get('torch')} (CUDA {h.get('torch_cuda')}, cuDNN {h.get('cudnn')}) | ultralytics {h.get('ultralytics')} | opencv {h.get('opencv')}")
    L.append(f" Video : {m.get('width')}x{m.get('height')} @ {m.get('fps')} fps | {m.get('total_frames')} frames in file | "
             f"YOLO imgsz={rep['config']['imgsz']} | precision={h.get('precision')}")
    L.append(f" Timing: cuda-sync={rep['config']['cuda_sync']} | warmup frames excluded={rep.get('warmup_frames_excluded')} | "
             f"profiler overhead ~{m.get('profiler_overhead_us_per_stage', 0):.0f} us/stage-call")

    n = rep.get("frames_counted", 0)
    if n == 0:
        L.append(" (koi frame process nahi hua)")
        return "\n".join(L)

    ft, rt = rep["frame_time_ms"], rep["realtime"]
    sec(f"1) OVERALL  ({n} frames counted)")
    L.append(f" mean {ft['mean']:.1f} ms/frame = {ft['fps_mean']:.2f} fps | p50 {ft['p50']:.1f} | p95 {ft['p95']:.1f} | "
             f"p99 {ft['p99']:.1f} | min {ft['min']:.1f} | max {ft['max']:.1f} ms")
    L.append(f" frame loop total {ft['loop_total_s']:.1f}s (wall {ft['loop_wall_s']:.1f}s) | whole run {m.get('run_total_s', 0):.1f}s")
    L.append(f" video {rt['video_fps']:.1f} fps -> pipeline {rt['processing_fps']:.2f} fps => {rt['realtime_factor']:.1f}x slower than real-time")
    if ft["warmup_frame_ms"]:
        L.append(" warmup frames (excluded): " + ", ".join(f"{x:.0f}ms" for x in ft["warmup_frame_ms"]))
    hist = rep["histogram"]
    mx = max(hist["counts"]) or 1
    L.append(" frame-time histogram:")
    for c, lo, hi in zip(hist["counts"], hist["edges_ms"][:-1], hist["edges_ms"][1:]):
        L.append(f"   {lo:8.0f}-{hi:8.0f} ms | {_bar(c / mx, 30)} {c}")

    sec("2) ONE-OFF COSTS  (model load / warmup / post-processing -- per video)")
    allone = {}
    for k, v in rep["startup_ms"].items():
        allone[k + "  [process startup]"] = v
    allone.update(rep["oneoff_ms"])
    for k, v in sorted(allone.items(), key=lambda kv: -kv[1]["ms"]):
        L.append(f" {k:<48} {v['ms']:>10.1f} ms  ({v['calls']} call{'s' if v['calls'] != 1 else ''})")
    if not allone:
        L.append(" -")

    sec("3) PER-FRAME STAGES  (sorted by total time; % = share of frame loop)")
    L.append(f" {'stage':<28}{'calls/f':>8}{'ms/call':>9}{'p95/call':>10}{'max/call':>10}{'ms/frame':>10}{'total s':>9}{'%loop':>7}{'GPU%':>6}  bar")
    for s, e in sorted(rep["stages"].items(), key=lambda kv: -kv[1]["total_ms"]):
        gu = f"{e['gpu_util_avg']:.0f}" if e["gpu_util_avg"] is not None else "-"
        L.append(f" {s:<28}{e['calls_per_frame']:>8.2f}{e['mean_ms_call']:>9.2f}{e['p95_ms_call']:>10.2f}{e['max_ms_call']:>10.2f}"
                 f"{e['mean_ms_frame']:>10.2f}{e['total_ms'] / 1000:>9.2f}{e['pct_loop']:>7.1f}{gu:>6}  {_bar(e['pct_loop'] / 100)}")
    u = rep["unaccounted"]
    L.append(f" {'(unaccounted)':<28}{'':>8}{'':>9}{'':>10}{'':>10}{u['mean_ms_frame']:>10.2f}{u['total_ms'] / 1000:>9.2f}{u['pct_loop']:>7.1f}{'':>6}  {_bar(u['pct_loop'] / 100)}")
    L.append(" (GPU% = NVML util jab ye stage active tha -- approx; low = stage CPU/IO par ruka hai)")

    sec("4) BY GROUP")
    for g, e in sorted(rep["groups"].items(), key=lambda kv: -kv[1]["total_ms"]):
        gu = f"GPU~{e['gpu_util_avg']:.0f}%" if e["gpu_util_avg"] is not None else ""
        L.append(f" {g:<10} {e['mean_ms_frame']:>9.2f} ms/frame {e['pct_loop']:>6.1f}%  {_bar(e['pct_loop'] / 100, 30)} {gu}")
    L.append("   det=YOLO detectors | reid=OSNet | track=DeepSORT+NMS | blink=indicator blink model | plate=plate detector | ocr=plate OCR")
    L.append("   logic=association/turn logic | io=disk/video read-write | draw=annotation")

    if rep["yolo_models"] or rep["model_info"]:
        sec("5) MODELS  (params / weights VRAM / YOLO internal split)")
        L.append(f" {'model':<18}{'params M':>9}{'VRAM MB':>9}{'call ms':>9}{'pre':>7}{'infer':>8}{'post':>7}{'py-ovh':>8}{'parse':>8}")
        for name, mi in rep["model_info"].items():
            y = rep["yolo_models"].get(name)
            if y:
                L.append(f" {name:<18}{mi.get('params_M', 0):>9}{mi.get('vram_MB', 0):>9}{y['model_call_ms']:>9.1f}{y['preprocess_ms']:>7.1f}"
                         f"{y['inference_ms']:>8.1f}{y['postprocess_ms']:>7.1f}{y['python_overhead_ms']:>8.1f}{y['parse_ms']:>8.1f}")
            else:
                L.append(f" {name:<18}{mi.get('params_M', 0):>9}{mi.get('vram_MB', 0):>9}")
        L.append(" (pre/infer/post = Ultralytics ka internal timing; py-ovh = model_call - (pre+infer+post); parse = box loop + print)")
        for k, v in rep["notes"].items():
            L.append(f" note {k}: mean {v['mean']:.2f}, max {v['max']:.0f} ({v['n']} samples)")

    sec("6) SCALING WITH NUMBER OF BIKES ON SCREEN")
    for b in rep["bike_scaling"]["buckets"]:
        L.append(f" {b['bikes']:>3} bike(s): {b['frames']:>6} frames | mean {b['mean_ms']:>8.1f} ms/frame")
    if rep["bike_scaling"]["ms_per_extra_bike"] is not None:
        L.append(f" linear fit: ~{rep['bike_scaling']['ms_per_extra_bike']:.1f} ms extra per additional bike")

    sec("7) SLOWEST FRAMES")
    for s in rep["slowest_frames"]:
        L.append(f" frame {s['frame']:>6}: {s['total_ms']:>8.1f} ms | bikes={s['bikes']} | culprit {s['culprit']} ({s['culprit_ms']:.1f} ms)")

    gm = rep.get("gpu_monitor")
    vr = rep.get("vram")
    if gm or vr:
        sec("8) GPU MONITOR (during frame loop)")
        if gm:
            f = lambda k, unit="": (f"avg {gm[k]['mean']:.0f}{unit} (min {gm[k]['min']:.0f}, max {gm[k]['max']:.0f})" if gm.get(k) else "n/a")
            L.append(f" util      : {f('util', '%')}   [{gm['samples']} samples via {gm['backend']}]")
            L.append(f" SM clock  : {f('sm_clock_mhz', ' MHz')}" + (f" = {gm['sm_clock_pct_of_max']:.0f}% of max" if gm.get("sm_clock_pct_of_max") else ""))
            L.append(f" power     : {f('power_w', ' W')} | temp: {f('temp_c', ' C')} | throttled samples: {gm.get('throttled_samples', 0)}")
        if vr:
            L.append(f" VRAM peak : allocated {vr['peak_allocated_mb']:.0f} MB | reserved {vr['peak_reserved_mb']:.0f} MB of {vr['total_mb']:.0f} MB")

    sec("9) PROJECTION  (is GPU par, is video jaisa content)")
    for lab, p in rep["projection"].items():
        L.append(f" {lab:<14} ({p['frames']} frames) -> ~{p['est_minutes']:.1f} min processing")
    w = rep.get("whatif_gpu")
    if w:
        L.append("")
        L.append(f" ROUGH what-if (measured on {w['measured_on']}; GPU-bound {w['gpu_bound_ms_frame']:.0f} ms/frame scaled by "
                 f"sqrt(TFLOPS ratio x BW ratio), CPU/IO {w['cpu_bound_ms_frame']:.0f} ms/frame unchanged):")
        for k, t in sorted(w["targets"].items(), key=lambda kv: kv[1]["est_ms_frame"]):
            L.append(f"   {k:<9} speed x{t['speed_ratio_vs_measured']:>5.2f} -> ~{t['est_ms_frame']:>7.1f} ms/frame ({t['est_fps']:.2f} fps)")
        L.append(" (estimate hai; asli number ke liye us GPU par run karke --compare use karo)")

    sec("10) VERDICT")
    for v in rep.get("verdict", []):
        L.append(" * " + v)
    L.append(hr)
    return "\n".join(L)


def compare_reports(paths):
    """Alag-alag GPUs ke profile JSON side-by-side compare karo."""
    if not paths:
        paths = sorted(glob.glob(os.path.join(LOG_DIR, PROFILE_SAVE_DIR_NAME, "*.json")))[-4:]
    reps = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            reps.append(json.load(f))
    if len(reps) < 1:
        print("Koi profile JSON nahi mila.")
        return ""
    labels = []
    for i, r in enumerate(reps):
        name = (r.get("hw") or {}).get("gpu_name", "?")
        labels.append(f"{name}" if [((x.get('hw') or {}).get('gpu_name')) for x in reps].count(name) == 1 else f"{name}#{i + 1}")
    cw = 22
    L = []
    L.append("=" * (34 + cw * len(reps)))
    L.append(" GPU COMPARISON  (baseline = first column)")
    L.append("=" * (34 + cw * len(reps)))
    L.append(f" {'':<32}" + "".join(f"{l[:cw - 2]:>{cw}}" for l in labels))

    def row(name, fn, fmt="{:>{w}.1f}"):
        cells = []
        for r in reps:
            try:
                cells.append(fmt.format(fn(r), w=cw))
            except Exception:
                cells.append(f"{'-':>{cw}}")
        L.append(f" {name:<32}" + "".join(cells))

    row("VRAM GB", lambda r: r["hw"]["gpu_vram_gb"])
    row("SM count", lambda r: r["hw"]["gpu_sm_count"], "{:>{w}d}")
    row("max SM clock MHz", lambda r: float(r["hw"]["gpu_max_sm_clock_mhz"]), "{:>{w}.0f}")
    row("video frames counted", lambda r: r["frames_counted"], "{:>{w}d}")
    row("mean ms/frame", lambda r: r["frame_time_ms"]["mean"])
    row("p95 ms/frame", lambda r: r["frame_time_ms"]["p95"])
    row("pipeline fps", lambda r: r["frame_time_ms"]["fps_mean"], "{:>{w}.2f}")
    row("x slower than realtime", lambda r: r["realtime"]["realtime_factor"])
    row("GPU util avg %", lambda r: r["gpu_monitor"]["util"]["mean"], "{:>{w}.0f}")
    row("SM clock avg MHz", lambda r: r["gpu_monitor"]["sm_clock_mhz"]["mean"], "{:>{w}.0f}")
    row("VRAM peak MB", lambda r: r["vram"]["peak_allocated_mb"], "{:>{w}.0f}")
    row("10-min video est. (min)", lambda r: r["projection"]["10 min video"]["est_minutes"])
    if len({(r.get("video"), (r.get("meta") or {}).get("width")) for r in reps}) > 1:
        L.append(" WARNING: alag video/resolution use hua -> comparison fair nahi hai.")

    L.append("")
    L.append(f" per-stage mean ms/frame   (Nx = baseline / this GPU, >1 matlab tez)")
    L.append(f" {'stage':<32}" + "".join(f"{l[:cw - 2]:>{cw}}" for l in labels))
    base = reps[0].get("stages", {})
    allst = sorted({s for r in reps for s in r.get("stages", {})}, key=lambda s: -base.get(s, {}).get("mean_ms_frame", 0))
    for s in allst:
        cells = []
        b = base.get(s, {}).get("mean_ms_frame")
        for i, r in enumerate(reps):
            v = r.get("stages", {}).get(s, {}).get("mean_ms_frame")
            if v is None:
                cells.append(f"{'-':>{cw}}")
            elif i == 0 or not b or v <= 0:
                cells.append(f"{v:>{cw}.2f}")
            else:
                cells.append(f"{f'{v:.2f} ({b / v:.2f}x)':>{cw}}")
        L.append(f" {s:<32}" + "".join(cells))
    L.append("")
    L.append(" per-group mean ms/frame")
    for g in sorted({g for r in reps for g in r.get("groups", {})}, key=lambda g: -reps[0].get("groups", {}).get(g, {}).get("mean_ms_frame", 0)):
        cells = []
        b = reps[0].get("groups", {}).get(g, {}).get("mean_ms_frame")
        for i, r in enumerate(reps):
            v = r.get("groups", {}).get(g, {}).get("mean_ms_frame")
            if v is None:
                cells.append(f"{'-':>{cw}}")
            elif i == 0 or not b or v <= 0:
                cells.append(f"{v:>{cw}.2f}")
            else:
                cells.append(f"{f'{v:.2f} ({b / v:.2f}x)':>{cw}}")
        L.append(f" {g:<32}" + "".join(cells))
    text = "\n".join(L)
    print(text)
    try:
        d = os.path.join(LOG_DIR, PROFILE_SAVE_DIR_NAME)
        os.makedirs(d, exist_ok=True)
        out = os.path.join(d, f"compare_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\nSaved: {out}")
    except Exception:
        pass
    return text




cfg = ModelConfig(**BLINK_MODEL_CONFIG)
def _load_blink():
    _m = build_model(cfg)
    return load_weights(_m, BLINK_WEIGHTS, device=device)
blink_detector = P.load_model("blink", _load_blink, input="seq of 64x64 indicator crops")


# ImageNet normalization (resnet18 pretrained isi se train hua tha)
_normalize = transforms.Compose([
    transforms.ToTensor(),   # (H, W, C) numpy -> (C, H, W) tensor, scaled to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def predict_from_track(model, track, slot_idx, device="cpu"):
    """
    Track's stored 64x64 indicator images (for ONE indicator slot — left or
    right, kept separate) -> single blink prediction for that slot.
    Returns (predicted_class, confidence) or None if that slot has no
    stored crops yet.
    """
    frames = track.getIndicatorImages(slot_idx)
    if not frames:
        return None

    with P.stage("blink.preprocess_h2d"):     # CPU normalize loop + host->GPU copy
        processed = []
        for frame in frames:
            frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            chw = _normalize(frame)          # (C, 64, 64), normalized
            hwc = chw.permute(1, 2, 0)       # (64, 64, C) -- model ye layout expect karta hai
            processed.append(hwc)

        seq = torch.stack(processed, dim=0).unsqueeze(0).to(device)   # (1, T, 64, 64, C)
        length = torch.tensor([seq.shape[1]], device=device)
    P.note("blink.seq_len", len(frames))

    with P.stage("blink.forward"):
        model.eval()
        logits = model(seq, length)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred].item()

    return pred, conf


def extract_indicator_crop(frame, indicator_box, size=(64, 64)):
    """
    Crop indicator from frame, resize to 64x64 and return numpy array.

    Parameters
    ----------
    frame : np.ndarray
        Original BGR image.
    indicator_box : list/tuple
        [confidence, x1, y1, x2, y2]
        or [x1, y1, x2, y2]
    size : tuple
        Output image size (default 64x64)

    Returns
    -------
    np.ndarray or None
        Shape: (64, 64, 3)
    """

    # Handle both formats
    if len(indicator_box) == 5:
        _, x1, y1, x2, y2 = indicator_box
    else:
        x1, y1, x2, y2 = indicator_box

    h, w = frame.shape[:2]

    # Clip coordinates
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.resize(crop, size)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # blink model expects RGB, frame is BGR (OpenCV)

    return crop.astype(np.uint8)


def build_osnet_encoder(model_path, model_name=OSNET_MODEL_NAME,
                         input_size=OSNET_INPUT_SIZE, device="cpu"):
    """
    Returns a callable `encoder(frame, boxes_tlwh) -> np.ndarray (N, feat_dim)`
    that matches the exact interface gdet.create_box_encoder(...) used to
    provide, so it drops straight into `features = encoder(frame, tlwhs)`
    with no other changes needed in run_video().

    Uses torchreid's FeatureExtractor, which handles resizing/normalization
    to match how the model was trained — just make sure `image_size` here
    matches what you trained OSNet at, or crops get distorted.
    """
    extractor = FeatureExtractor(
        model_name=model_name,
        model_path=model_path,
        image_size=input_size,   # (H, W)
        device=device,
    )

    def encoder(frame, boxes_tlwh):
        if len(boxes_tlwh) == 0:
            return np.zeros((0, 512), dtype=np.float32)  # osnet_x1_0 default feat dim

        h_frame, w_frame = frame.shape[:2]
        crops = []
        for (x, y, w, h) in boxes_tlwh:
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w_frame, int(x + w)), min(h_frame, int(y + h))
            if x2 <= x1 or y2 <= y1:
                # degenerate box (can happen if a tlwh comes in clipped to
                # nothing) — feed a black patch so the batch stays aligned
                # with boxes_tlwh instead of silently dropping an index
                crop = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
            else:
                crop = frame[y1:y2, x1:x2]
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        P.metric("n_reid_crops", len(crops))
        with torch.no_grad():
            with P.stage("reid.extract"):     # torchreid: CPU resize/PIL + H2D + OSNet forward
                feats = extractor(crops)                       # (N, feat_dim) torch tensor
            with P.stage("reid.postproc_d2h"):
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)
                out = feats.cpu().numpy().astype(np.float32)

        return out

    encoder.extractor = extractor
    return encoder


# ----------------- Optical flow: removed -----------------
# RAFT (deep-learning optical flow) was pulled out entirely -- its
# correlation volume was OOM-ing the GPU at video-native resolution on top
# of YOLO/OSNet/the blink model already resident in VRAM. bike_flow_stats()
# below already treats flow=None as "no flow data" and returns all zeros,
# so run_video() just calls it with flow=None now -- setFlow()/CSV
# output/etc. downstream keep working unchanged, they just always get
# (0, 0, 0, 0) for the flow fields. If flow is needed again later, a
# lightweight classical method (e.g. cv2.calcOpticalFlowFarneback, CPU-only,
# no GPU memory pressure) is a much safer drop-in than another
# deep-learning flow-net on a shared T4.


def bike_flow_stats(flow, bbox):
    """
    Average optical-flow vector inside a single bike's bbox for this frame.
    Returns (dx, dy, magnitude, angle_deg) — all zero if flow is unavailable
    (first frame) or the box is degenerate.
    """
    if flow is None:
        return 0.0, 0.0, 0.0, 0.0

    h, w = flow.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0, 0.0

    patch = flow[y1:y2, x1:x2]
    dx = float(np.mean(patch[..., 0]))
    dy = float(np.mean(patch[..., 1]))
    magnitude = float(np.hypot(dx, dy))
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return dx, dy, magnitude, angle


def xyxy_to_tlwh(box):
    # box = [x1, y1, x2, y2]
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return [int(x1), int(y1), int(w), int(h)]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def is_box_inside(box1, box2, method="center"):
    """
    Check if box2 is inside box1.

    Parameters:
        box1: list or tuple [x1, y1, x2, y2]  # parent box
        box2: list or tuple [x1, y1, x2, y2]  # child box
        method: "center" (default), "full", or "iou"
    
    Returns:
        True if box2 is considered inside box1, else False
    """

    x1, y1, x2, y2 = box1
    bx1, by1, bx2, by2 = box2

    if method == "center":
        # Check center point of box2
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        return x1 <= cx <= x2 and y1 <= cy <= y2

    elif method == "full":
        # Check full containment
        return x1 <= bx1 and y1 <= by1 and x2 >= bx2 and y2 >= by2

    elif method == "iou":
        # Check IoU threshold (partial overlap)
        inter_x1 = max(x1, bx1)
        inter_y1 = max(y1, by1)
        inter_x2 = min(x2, bx2)
        inter_y2 = min(y2, by2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (bx2 - bx1) * (by2 - by1)
        iou = inter_area / (area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0

        return iou > 0.1  # threshold adjustable

    else:
        raise ValueError("Method must be 'center', 'full', or 'iou'")

def _zone_polygon_points(zone):
    """Return an (N,2) int32 array of vertices for a polygon/rectangle zone.
    Rectangles given as 2 corner points get expanded to 4; anything else
    (>=3 points, arbitrary polygon) passes through as-is. Not used for
    circles, which are rasterized directly.
    """
    pts = zone["points"]
    if len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        x1, x2 = sorted((float(x1), float(x2)))
        y1, y2 = sorted((float(y1), float(y2)))
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return np.array(pts, dtype=np.int32)


def build_zone_mask(zones, width, height):
    """Rasterize every configured turn zone — any mix of polygon, rectangle,
    or circle, any count — into a single binary (H, W) mask where a pixel is
    1 if it falls inside ANY zone. Built once per video (zones are treated
    as static for the whole clip), so per-frame overlap checks below are
    just a cheap crop + count against this mask.

    Returns None if `zones` is empty, so callers can skip zone-based turn
    detection entirely when no zones are configured (the "no polygon" case).
    """
    if not zones:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    for zone in zones:
        ztype = zone.get("type", "polygon").lower()
        if ztype == "circle":
            cx, cy = map(int, zone["center"])
            r = int(zone["radius"])
            cv2.circle(mask, (cx, cy), r, 1, thickness=-1)
        elif ztype in ("polygon", "rectangle"):
            pts_arr = _zone_polygon_points(zone)
            cv2.fillPoly(mask, [pts_arr], 1)
        else:
            raise ValueError(f"Unknown turn zone type: {ztype!r} (expected 'polygon', 'rectangle', or 'circle')")
    return mask


def zone_overlap_ratio(mask, bbox):
    """Fraction (0.0-1.0) of `bbox` [x1,y1,x2,y2] that falls inside ANY
    configured turn zone, per the mask from build_zone_mask(). Returns 0.0
    if there's no mask (no zones configured) or the box is degenerate/fully
    off-frame.
    """
    if mask is None:
        return 0.0

    h, w = mask.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    box_area = (x2 - x1) * (y2 - y1)
    inside = int(np.count_nonzero(mask[y1:y2, x1:x2]))
    return inside / box_area if box_area > 0 else 0.0


def draw_zones(frame, zones, color=TURN_ZONE_DRAW_COLOR, thickness=TURN_ZONE_DRAW_THICKNESS):
    """Draw every configured turn zone onto `frame` — any mix of polygon,
    rectangle, or circle, any count — so the output video shows exactly what
    bikes are being checked against. No-op if `zones` is empty.
    """
    for zone in zones:
        ztype = zone.get("type", "polygon").lower()
        if ztype == "circle":
            cx, cy = map(int, zone["center"])
            r = int(zone["radius"])
            cv2.circle(frame, (cx, cy), r, color, thickness)
        elif ztype in ("polygon", "rectangle"):
            pts_arr = _zone_polygon_points(zone).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_arr], isClosed=True, color=color, thickness=thickness)
        else:
            raise ValueError(f"Unknown turn zone type: {ztype!r} (expected 'polygon', 'rectangle', or 'circle')")


def expand_box(x1,y1,x2,y2,  w, h, pad_ratio=0.30):
    bw = x2-x1
    bh = y2-y1

    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)

    x1 = max(0, x1-pad_w)
    y1 = max(0, y1-pad_h)
    x2 = min(w, x2+pad_w)
    y2 = min(h, y2+pad_h)

    return x1,y1,x2,y2


def getDetections(model, frame , detect , confidence = MIN_CONFIDENCE, tag = "det.unknown"):
        try:
            with P.stage(f"{tag}.model_call"):        # preprocess + GPU inference + NMS, CUDA-synced
                results = model(frame, imgsz=INFERENCE_IMAGE_SIZE)
            try:
                P.note_ult(tag, results[0].speed)     # Ultralytics' own pre/infer/post ms
            except Exception:
                pass
        except Exception:
            log_exception(f"YOLO inference failed for detect={detect}")
            return [], []

        detections = []
        class_name = []
        with P.stage(f"{tag}.parse"):                 # python box loop + .cpu().numpy() + print()
            try:
                res = results[0]
                if hasattr(res, "boxes"):
                    for box in res.boxes:
                        try:
                            cls_id= int(box.cls[0]) if hasattr(box.cls, "__getitem__") else int(box.cls)
                            name = res.names[cls_id]
                            print(f'ClassName : give: {detect} -> found {name}')
                            mx1, my1, mx2, my2 = box.xyxy[0].cpu().numpy()
                            mx1, my1, mx2, my2 = int(mx1), int(my1), int(mx2), int(my2)
                            conf = float(box.conf[0].cpu().numpy()) if hasattr(box.conf, "__getitem__") else float(box.conf)
                            if conf > confidence and name in detect:
                                detections.append([conf, mx1, my1, mx2, my2])
                                class_name.append(name)
                        except Exception:
                            log_exception(f"parsing one YOLO box for detect={detect}")
                            continue
            except Exception:
                log_exception(f"parsing YOLO results for detect={detect}")

        return detections , class_name


def active_tracks(tracker):
    """
    Tracks that are confirmed (passed n_init) AND were actually matched to a
    detection this frame (time_since_update == 0).

    Iterating tracker.tracks directly (as the original code did) also pulls
    in tentative tracks (unconfirmed, can vanish next frame) and stale/ghost
    tracks that Deep SORT is still coasting on Kalman prediction alone for up
    to max_age frames after last seeing them. Drawing/associating against
    those is what produces "duplicate" boxes on a single physical object
    when a fresh track spawns for it while the old ghost track is still
    being rendered.
    """
    return [t for t in tracker.tracks if t.is_confirmed() and t.time_since_update == 0]


def best_match_track(tracks, box, method="center"):
    """
    Return the single track whose box has the highest IoU with `box` among
    those where is_box_inside(...) also holds, or None.

    The original code assigned a mirror/indicator/orientation detection to
    EVERY track that satisfied is_box_inside(). When two tracked bikes
    overlap (common in traffic), that lets one detection get attached to
    more than one track -> duplicate annotations that look like ID mixups.
    Picking a single best match fixes that.
    """
    best_track, best_iou = None, 0.0
    for t in tracks:
        t_box = t.to_tlbr()
        if not is_box_inside(t_box, box, method=method):
            continue
        iou = compute_iou(t_box, box)
        if iou >= best_iou:
            best_iou = iou
            best_track = t
    return best_track


def _save_bike_snapshot(track, images_dir):
    """Write (or overwrite) this bike's PUBLIC annotated frame to disk
    RIGHT NOW, using whatever track.full_frame_image / track.full_frame_box
    currently hold:
        images_dir/<track_id>/annotated_frame.jpg   (full frame + box —
            this is the final artifact a human/frontend sees)

    Called every time update_best_image() reports a NEW best (bigger) crop
    — i.e. continuously, throughout video processing, not just once when
    the track ends — so the file keeps getting overwritten by whichever
    frame gave the biggest view of this bike so far. Also called once more
    from finalize_track() as a safety net.

    The bike's plate_crop.jpg is a SEPARATE, independent file — see
    _save_plate_snapshot() below, driven by its own "biggest plate box
    seen so far" tracking (best_plate_area_by_track in run_video()), not
    tied to whichever frame happened to have the biggest overall bike crop.

    Returns the annotated_frame.jpg path, or None if there was nothing to
    save.
    """
    if images_dir is None or track.best_image is None:
        return None
    try:
        bike_dir = os.path.join(images_dir, str(track.track_id))
        os.makedirs(bike_dir, exist_ok=True)

        if track.full_frame_image is not None:
            annotated = track.full_frame_image.copy()
            if track.full_frame_box is not None:
                x1, y1, x2, y2 = track.full_frame_box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"ID:{track.track_id}", (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            annotated_frame_path = os.path.join(bike_dir, "annotated_frame.jpg")
            cv2.imwrite(annotated_frame_path, annotated)
            return annotated_frame_path

        return None
    except Exception as e:
        print(f"Track {track.track_id}: failed to save snapshot image ({e}).")
        return None


def _save_plate_snapshot(track_id, plate_crop_rgb, images_dir):
    """Write (or overwrite) this bike's PUBLIC plate crop to disk RIGHT NOW:
        images_dir/<track_id>/plate_crop.jpg

    Called every frame the plate DETECTOR (running live, inside the main
    loop — see run_video()) finds a bigger/clearer plate box for this bike
    than any seen before that video. NO OCR happens here or anywhere in the
    frame loop — this only ever saves the located CROP. Reading the actual
    numbers off of it is the post-loop pass's job, at the very end of
    run_video(), once every frame has been processed — see
    read_plate_text_from_image() in plate_detector.py.

    Parameters
    ----------
    plate_crop_rgb : np.ndarray
        RGB image. detect_plates_in_bike_crop() converts its input to RGB
        internally before slicing the plate out of it, so what it returns
        is RGB, not the BGR order cv2 normally uses — flip it back here
        before cv2.imwrite, or the saved jpg's colors come out swapped.

    Returns the plate_crop.jpg path, or None if there was nothing to save.
    """
    if images_dir is None or plate_crop_rgb is None or plate_crop_rgb.size == 0:
        return None
    try:
        bike_dir = os.path.join(images_dir, str(track_id))
        os.makedirs(bike_dir, exist_ok=True)
        plate_crop_bgr = cv2.cvtColor(plate_crop_rgb, cv2.COLOR_RGB2BGR)
        plate_crop_path = os.path.join(bike_dir, "plate_crop.jpg")
        cv2.imwrite(plate_crop_path, plate_crop_bgr)
        return plate_crop_path
    except Exception as e:
        print(f"Track {track_id}: failed to save plate crop ({e}).")
        return None


def finalize_track(track, records, images_dir=None):
    """Called exactly once per bike, the moment its track is gone for good
    — either DeepSORT actually deleted it (lost for too many frames), or
    the video ended while it was still alive.

    Both plate DETECTION and OCR already happened live, every frame this
    bike was on screen, in the main loop (see run_video()'s plate
    DETECTOR + live-OCR block) — so by the time this runs,
    images_dir/<track_id>/plate_crop.jpg already holds the biggest/
    clearest plate crop ever found for this bike, and track.plate_text
    already holds its OCR reading, if one was ever successfully obtained
    (track.plate_locked True). Nothing else needs to run here to resolve
    the plate — this function just carries that result into `records`.

    So this function:
      1. Re-saves this bike's PUBLIC annotated frame via
         _save_bike_snapshot() — images_dir/<track_id>/annotated_frame.jpg
         — as a safety net, in case the frame loop's own continuous saving
         never fired for this track.
      2. Appends one record — everything the final CSV row needs, including
         plate_number straight from track.plate_text (empty string if this
         bike's plate was never successfully read) — to `records`, a list
         shared across the whole run_video() call. Any track whose plate
         is still unresolved here gets one more OCR attempt in the
         post-loop fallback pass near the end of run_video(), reading
         straight off its saved plate_crop.jpg.

    Tracks that never produced a large-enough crop (see
    MIN_BIKE_AREA_FOR_BEST_IMAGE / Track.update_best_image) are skipped
    entirely and never get a record — a track that was always tiny was
    most likely a noisy/false detection, never a bike close enough to
    read a plate from, so it isn't worth a row.
    """
    if track.best_image is None:
        print(f"Track {track.track_id}: no large-enough crop ever captured — treating as noise, skipping.")
        return

    annotated_frame_path = _save_bike_snapshot(track, images_dir)
    if annotated_frame_path:
        print(f"Track {track.track_id}: snapshot image -> {annotated_frame_path}")

    records.append({
        "track_id": track.track_id,
        "plate_number": track.plate_text or "",
        # True if the plate DETECTOR (YOLO) ever found a plate box on this
        # bike at least once, regardless of whether OCR could read it —
        # see Track.plate_detected in track.py.
        "plate_detected": bool(getattr(track, "plate_detected", False)),
        "mirror_seen_both": track.mirror_seen_both,
        "indicator_seen_both": track.indicator_seen_both,
        "ever_turned": track.ever_turned,
        "ever_signaled_while_turning": track.ever_signaled_while_turning,
    })

    print(f"Track {track.track_id} finalized — plate={track.plate_text!r} "
          f"(locked={track.plate_locked}, detected={getattr(track, 'plate_detected', False)}), "
          f"both_mirrors={track.mirror_seen_both}, "
          f"both_indicators={track.indicator_seen_both}, turned={track.ever_turned}, "
          f"signaled_while_turning={track.ever_signaled_while_turning}")


def draw_box(image, coords, label="Box", color=(0,255,0)):
    x1, y1, x2, y2 = map(int, coords)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def _safe_slug(name, fallback="video"):
    """Filesystem-safe base name (extension stripped), used for BOTH the
    public snapshot folder and the summary CSV filename — so the same
    original video name always maps to the same on-disk location, and
    re-uploading a video with that name overwrites its previous run
    instead of piling up next to it."""
    base = os.path.splitext(name)[0]
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._")
    return base or fallback


def run_video(video_path, output_path=None, display_name=None):
    """
    Parameters
    ----------
    video_path : str
        Actual file on disk to read frames from (can be a
        timestamp-prefixed filename so concurrent uploads never collide).
    output_path : str, optional
        Root folder this video's own output folder (named after
        video_base) gets created under. Everything this run produces —
        the annotated video, the summary CSV, and the per-bike bikes/
        folder — lives together inside <output_path>/<video_base>/.
    display_name : str, optional
        The video's ORIGINAL name (e.g. exactly what the user uploaded).
        Used to name that per-video output folder, so re-uploading a video
        with this same name OVERRIDES its previous output folder instead
        of creating a separate copy. Falls back to video_path's own
        filename when not given.

    Returns
    -------
    dict with:
        "output_video_path" : the final browser-playable .mp4 path
        "summary_csv_path"  : the per-bike summary CSV path (written once,
                               at the very end. Plate detection AND OCR
                               both already ran live, every frame, during
                               the main loop — see finalize_track() and
                               the plate DETECTOR + live-OCR block)
        "video_images_dir"  : the public per-bike "bikes" folder, one
                               subfolder per track_id
        "video_dir"         : the per-video root folder holding all of the
                               above (video, CSV, bikes/)
        "video_base"        : the sanitized base name used for video_dir
    """
    vidName = video_path.split("/")[-1]

    # Base name for this video's own output folder + files. Prefers the
    # ORIGINAL video name (display_name) so repeat uploads of the same
    # video overwrite the same location; falls back to the on-disk name
    # (e.g. when run_video() is called directly, see __main__ below).
    video_base = _safe_slug(display_name if display_name else vidName)
    P.begin_run(video_base)

    # Everything this video produces now lives together under ONE folder:
    #
    #   <output_root>/<video_base>/
    #       annotatedvideo.mp4
    #       <video_base>_bikes.csv
    #       bikes/<track_id>/annotated_frame.jpg
    #       bikes/<track_id>/plate_crop.jpg
    #
    # `output_path` (the folder argument) is the root that <video_base>/
    # gets created under; PUBLIC_OUTPUT_DIR is only a fallback for when
    # run_video() is called without one (see __main__ below).
    #
    # If this video name was processed before, wipe its whole folder first
    # — an override, not a merge, so nothing stale (old bike folders, old
    # CSV, old video) from a previous run can ever linger into this one.
    output_root = output_path if output_path is not None else PUBLIC_OUTPUT_DIR
    video_dir = os.path.join(output_root, video_base)
    if os.path.isdir(video_dir):
        try:
            with P.stage("setup.clear_previous_output"):
                shutil.rmtree(video_dir)
        except Exception:
            log_exception(f"clearing previous output folder for video_base={video_base!r}")
    os.makedirs(video_dir, exist_ok=True)

    output_path = os.path.join(video_dir, "annotatedvideo.mp4")

    # The actual per-bike summary CSV is only written ONCE, at the very end
    # of this function — see near the bottom. No live per-row writes. Both
    # plate detection AND OCR already ran live, every bike, every frame, in
    # the main loop below.
    summary_csv_path = os.path.join(video_dir, video_base + "_bikes.csv")

    # Public per-bike snapshot folder — one subfolder per bike (named by
    # track_id), each holding that bike's annotated_frame.jpg and
    # plate_crop.jpg, both written continuously during the loop as bigger/
    # clearer crops are found.
    video_images_dir = os.path.join(video_dir, "bikes")
    os.makedirs(video_images_dir, exist_ok=True)

    # Every Track object we've ever seen this video, keyed by track_id. Kept
    # alive here even after DeepSORT drops a deleted track from
    # tracker.tracks, so we can still read its accumulated state
    # (best_image, plate_text, mirror/indicator facts, turn facts) to
    # finalize it.
    known_tracks = {}

    # Filled in by finalize_track() as each bike's track ends — one dict per
    # bike with everything the CSV needs, including plate_number (from
    # track.plate_text, resolved live during the loop for almost every
    # bike; any still-empty ones get one more try in the post-loop fallback
    # OCR pass near the end of this function).
    records = []

    # track_id -> area (pixels, w*h) of the biggest plate box the plate
    # DETECTOR has found for that bike so far, this video. Lives for the
    # same lifetime as known_tracks (a run_video()-wide dict, not reset per
    # frame) so a bike's plate_crop.jpg only ever gets overwritten by a
    # BIGGER/clearer plate box, never a smaller one — see the plate
    # DETECTOR block in the main loop below and _save_plate_snapshot().
    best_plate_area_by_track = {}

    yolo = P.load_model("det.bike", lambda: YOLO(str(YOLO_BIKE_MODEL_PATH)).to(device), weights=str(YOLO_BIKE_MODEL_PATH), imgsz=INFERENCE_IMAGE_SIZE)
    mirror_detector = P.load_model("det.mirror", lambda: YOLO(str(MIRROR_MODEL_PATH)).to(device), weights=str(MIRROR_MODEL_PATH), imgsz=INFERENCE_IMAGE_SIZE)
    indicator_detector = P.load_model("det.indicator", lambda: YOLO(str(INDICATOR_MODEL_PATH)).to(device), weights=str(INDICATOR_MODEL_PATH), imgsz=INFERENCE_IMAGE_SIZE)
    orientation_detector = P.load_model("det.orientation", lambda: YOLO(str(ORIENTATION_MODEL_PATH)).to(device), weights=str(ORIENTATION_MODEL_PATH), imgsz=INFERENCE_IMAGE_SIZE)

    # Plate DETECTOR (YOLO) — preloaded up front, right alongside the other
    # per-frame detectors above, since it runs every frame inside the main
    # loop (see the plate DETECTOR block below).
    try:
        with P.stage("load.warmup_plate_detector"):
            warm_up_plate_detector()
    except Exception:
        log_exception("warm_up_plate_detector() before main loop")

    # OCR (EasyOCR) — also preloaded up front now. EasyOCR is PyTorch-based
    # so, unlike PaddleOCR, it shares this process's existing CUDA context
    # fine alongside YOLO/OSNet/the blink model — no need to keep it out of
    # the main loop or unload everything else first. OCR itself is still
    # called sparingly inside the loop (only on a NEW largest plate crop
    # for a track that hasn't been read yet — see the plate DETECTOR block
    # below and Track.plate_locked in track.py), not on every frame.
    try:
        with P.stage("load.warmup_ocr"):
            warm_up_ocr()
    except Exception:
        log_exception("warm_up_ocr() before main loop")

    # Init DeepSORT
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
    tracker = Tracker(metric, max_iou_distance=MAX_IOU_DISTANCE, max_age=MAX_AGE, n_init=N_INIT)

    encoder = P.load_model("reid.osnet", lambda: build_osnet_encoder(REID_WEIGHTS, device=device),
                           weights=REID_WEIGHTS, input=list(OSNET_INPUT_SIZE))

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    P.set_run_meta(width=width, height=height, fps=fps, total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                   video_path=video_path)

    # OpenCV's mp4v encoder is reliable to write to, but the resulting file
    # uses a codec browsers can't decode (only desktop players like VLC/WMP
    # can). So we write to a temp file with mp4v as before, then re-encode
    # it to H.264 (browser-playable) as a final step below, and delete the
    # temp file. `output_path` stays the final, browser-compatible filename.
    raw_output_path = None
    writer = None
    if output_path:
        raw_output_path = output_path.replace(".mp4", "_raw_temp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    # Built once — zones are static for the whole clip. None if TURN_ZONES
    # is empty, in which case zone_overlap_ratio() always returns 0.0 and
    # the pipeline runs on angle-based turn detection alone (unchanged
    # behavior for videos with no zones configured).
    zone_mask = build_zone_mask(TURN_ZONES, width, height)

    trails = {}
    frame_idx = 0
    t0 = time.time()

    P.start_loop()
    while True:
        P.begin_frame(frame_idx + 1)
        try:
            with P.stage("io.cap_read"):          # video decode
                ret, frame = cap.read()
        except Exception:
            log_exception(f"cap.read() at frame_idx={frame_idx}")
            P.abort_frame()
            break
        if not ret:
            P.abort_frame()
            break
        frame_idx += 1
        # Heartbeat — flushed immediately. If the process dies with no
        # traceback, the LAST of these lines printed tells you exactly
        # which frame it died on.
        print(f"[FRAME {frame_idx}] --- start ---", flush=True)
        logger.debug(f"[FRAME {frame_idx}] start")

        flow = None  # optical flow removed (was RAFT) -- bike_flow_stats() returns zeros for this

        try:
            motorcycle_boxes ,_ = getDetections(yolo , frame , ['motorcycle'], tag="det.bike");
        except Exception:
            log_exception(f"getDetections(yolo/motorcycle) at frame_idx={frame_idx}")
            motorcycle_boxes = []

        detections = []
        P.metric("n_bike_dets", len(motorcycle_boxes))
        try:
            if motorcycle_boxes:
                tlwhs = [xyxy_to_tlwh(p[1:]) for p in motorcycle_boxes]
                confidences = [p[0] for p in motorcycle_boxes]
                features = encoder(frame, tlwhs)

                # --- NMS on raw detections BEFORE they reach the tracker -------
                # Without this, two overlapping YOLO boxes on the same bike both
                # get fed to tracker.update() and Deep SORT can spawn a second
                # track for the same physical object -> duplicate IDs.
                boxes_arr = np.array(tlwhs, dtype=np.float32)
                scores_arr = np.array(confidences, dtype=np.float32)
                with P.stage("track.nms"):
                    keep = preprocessing.non_max_suppression(boxes_arr, NMS_MAX_OVERLAP, scores_arr)

                for i in keep:
                    detections.append(Detection(tlwhs[i], confidences[i], features[i]))
        except Exception:
            log_exception(f"OSNet encoder / NMS at frame_idx={frame_idx}")
            detections = []

        try:
            with P.stage("track.deepsort_predict"):
                tracker.predict()
            with P.stage("track.deepsort_update"):    # cosine-distance matching + Kalman update
                tracker.update(detections)
        except Exception:
            log_exception(f"tracker.predict()/update() at frame_idx={frame_idx}")
            continue

        # DeepSORT's Tracker.update() drops a track from tracker.tracks the
        # instant it's marked Deleted (lost for too many frames). Any
        # track_id that was in known_tracks last frame but is missing from
        # tracker.tracks now was JUST deleted -> this is the one moment we
        # finalize it (resolve its plate from the stored largest image,
        # write its one summary CSV row) and drop our own reference too.
        current_ids = {t.track_id for t in tracker.tracks}
        for tid in list(known_tracks.keys()):
            if tid not in current_ids:
                try:
                    with P.stage("io.finalize_track"):
                        finalize_track(known_tracks[tid], records, video_images_dir)
                except Exception:
                    log_exception(f"finalize_track(track_id={tid}) at frame_idx={frame_idx}")
                del known_tracks[tid]
        for t in tracker.tracks:
            known_tracks[t.track_id] = t

        try:
            mirrors,_  = getDetections(mirror_detector, frame , ['mirror'], tag="det.mirror");
        except Exception:
            log_exception(f"getDetections(mirror_detector) at frame_idx={frame_idx}")
            mirrors = []
        try:
            indicators,_  = getDetections(indicator_detector, frame , ['indicator'], tag="det.indicator");
        except Exception:
            log_exception(f"getDetections(indicator_detector) at frame_idx={frame_idx}")
            indicators = []
        try:
            orientation_boxes, orientation_classes = getDetections(
                orientation_detector,
                frame,
                ['front-back', 'side'],
                tag="det.orientation"
            )
        except Exception:
            log_exception(f"getDetections(orientation_detector) at frame_idx={frame_idx}")
            orientation_boxes, orientation_classes = [], []

        # NOTE: mirrors and indicators are no longer wiped every frame here.
        # Both now persist across missed frames on purpose (see
        # Track.update_mirrors() / Track.update_indicators()) — a mirror or
        # indicator YOLO fails to find this frame keeps its last-known box,
        # shifted forward by the bike's own motion, instead of going blank.

        # Only associate/draw against tracks that are confirmed AND were
        # actually matched to a detection this frame. Using tracker.tracks
        # directly pulls in tentative and stale/ghost tracks, which is the
        # other main source of "duplicate" boxes and mis-assigned
        # mirror/indicator/orientation detections.
        live_tracks = active_tracks(tracker)
        P.metric("n_live_bikes", len(live_tracks))
        P.metric("n_mirror_dets", len(mirrors))
        P.metric("n_indicator_dets", len(indicators))

        # Update trajectories / turn detection FIRST, so bike.is_turning
        # below reflects THIS frame rather than lagging by one frame (as in
        # the original ordering, where detect_turn() ran after is_turning
        # was already consumed for the indicator/blink check).
        turn_results = {}
        for bike in live_tracks:
            try:
                with P.stage("logic.trajectory_turn"):
                    bike.setTrajectories(bike.to_tlbr())
                    turn_results[bike.track_id] = bike.detect_turn()

                # Turn-zone check: works alongside the angle-based detector
                # above, not instead of it — either one flagging a turn is
                # enough. With TURN_ZONES empty, zone_mask is None and this is a
                # no-op (ratio 0.0, is_turning left exactly as detect_turn() set
                # it), so this is safe to leave on for every video.
                with P.stage("logic.turn_zone"):
                    ratio = zone_overlap_ratio(zone_mask, bike.to_tlbr())
                bike.zone_overlap_ratio = ratio
                if ratio >= TURN_ZONE_OVERLAP_THRESHOLD:
                    bike.is_turning = True
            except Exception:
                log_exception(f"trajectory/turn update track_id={bike.track_id} at frame_idx={frame_idx}")

        for box, orientation in zip(orientation_boxes, orientation_classes):
            try:
                bbox = box[1:]  # x1, y1, x2, y2
                with P.stage("logic.orientation_assign"):
                    bike = best_match_track(live_tracks, bbox, method="iou")
                    if bike is not None:
                        bike.setOrientation(orientation)
            except Exception:
                log_exception(f"orientation assignment at frame_idx={frame_idx}")

        # Group this frame's mirror detections by the track they matched,
        # for the same reason indicators are grouped below: update_mirrors()
        # needs to see both of a bike's mirror boxes together (0, 1, or 2)
        # to match them against the two persistent slots correctly.
        mirrors_by_track = defaultdict(list)
        for mirror in mirrors:
            try:
                with P.stage("logic.mirror_assign"):
                    bike = best_match_track(live_tracks, mirror[1:], method="center")
                if bike is not None:
                    mirrors_by_track[bike.track_id].append(mirror)
            except Exception:
                log_exception(f"mirror best_match_track at frame_idx={frame_idx}")

        for bike in live_tracks:
            try:
                dets = mirrors_by_track.get(bike.track_id, [])

                traj = bike.getAllTrajectoriesOfX()
                if len(traj) >= 2:
                    move_dx = traj[-1][0] - traj[-2][0]
                    move_dy = traj[-1][1] - traj[-2][1]
                else:
                    move_dx, move_dy = 0.0, 0.0

                with P.stage("logic.mirror_update"):
                    bike.update_mirrors(dets, move_dx, move_dy,
                                         min_hits_for_both=MIN_SLOT_HITS_FOR_BOTH)
            except Exception:
                log_exception(f"update_mirrors track_id={bike.track_id} at frame_idx={frame_idx}")

        # Group this frame's indicator detections by the track they matched,
        # so both indicators on a bike (0, 1, or 2 boxes) are handed to
        # update_indicators() together — the nearest-neighbor slot matching
        # needs to see them as a set, not one at a time, or it can't tell
        # which detection is "the one that was already left" vs "right".
        indicators_by_track = defaultdict(list)
        for indicator in indicators:
            try:
                with P.stage("logic.indicator_assign"):
                    bike = best_match_track(live_tracks, indicator[1:], method="center")
                if bike is not None:
                    indicators_by_track[bike.track_id].append(indicator)
            except Exception:
                log_exception(f"indicator best_match_track at frame_idx={frame_idx}")

        for bike in live_tracks:
            try:
                dets = indicators_by_track.get(bike.track_id, [])

                # How far the bike itself moved this frame, so a slot that
                # misses detection this frame can still shift forward with the
                # bike instead of freezing or vanishing.
                traj = bike.getAllTrajectoriesOfX()
                if len(traj) >= 2:
                    move_dx = traj[-1][0] - traj[-2][0]
                    move_dy = traj[-1][1] - traj[-2][1]
                else:
                    move_dx, move_dy = 0.0, 0.0

                with P.stage("logic.indicator_update"):
                    bike.update_indicators(dets, move_dx, move_dy,
                                            min_hits_for_both=MIN_SLOT_HITS_FOR_BOTH)
            except Exception:
                log_exception(f"update_indicators track_id={bike.track_id} at frame_idx={frame_idx}")
                continue

            if not bike.is_turning:
                continue

            any_signal = False
            for slot_idx, slot in enumerate(bike.indicator_slots):
                if slot["cord"] is None:
                    continue

                try:
                    with P.stage("blink.crop"):
                        croped = extract_indicator_crop(frame, slot["cord"])
                    if croped is not None:
                        # Each slot keeps its own crop history, so the blink
                        # model is trained/queried per-indicator, not on a mix
                        # of left+right frames.
                        bike.setIndicatorImage(slot_idx, croped)

                    result = predict_from_track(blink_detector, bike, slot_idx, device=device)
                    if result:
                        pred_class, confidence = result
                        slot["is_signal"] = True if pred_class == 0 else False
                        print(f"Track {bike.track_id} slot {slot_idx} prediction: {pred_class} ({confidence:.2%})", flush=True)
                except Exception:
                    log_exception(f"blink prediction track_id={bike.track_id} slot={slot_idx} at frame_idx={frame_idx}")
                any_signal = any_signal or slot["is_signal"]

            # Bike-level flag: on if EITHER indicator is judged to be
            # blinking (normal turning uses just one side; both only for
            # hazards, which still counts as "signaling").
            bike.is_signal = any_signal

        # Centralized color rule — applies to EVERY live track (not just
        # the ones that were turning), so straight bikes get green too.
        # Same rule regardless of whether is_turning came from the
        # angle-based detector, a turn-zone overlap, or both:
        #   not turning            -> green
        #   turning, no indicator  -> red
        #   turning, indicator ON  -> black
        for bike in live_tracks:
            try:
                with P.stage("logic.color_mark"):
                    bike.update_box_color()
                # Roll this frame's is_turning/is_signal into the sticky,
                # track-lifetime "ever turned" / "ever signaled while turning"
                # facts used in the final CSV summary.
                with P.stage("logic.color_mark"):
                    bike.mark_turn_frame()
            except Exception:
                log_exception(f"update_box_color/mark_turn_frame track_id={bike.track_id} at frame_idx={frame_idx}")

        # Keep each live bike's single largest-ever crop, and — the moment
        # a NEW bigger crop wins — overwrite that bike's annotated_frame.jpg
        # on disk right away. This runs continuously for as long as the
        # video is being processed, so the saved image only ever gets
        # replaced by a BIGGER one, never a smaller one. Small/far/blurry
        # boxes are ignored (see MIN_BIKE_AREA_FOR_BEST_IMAGE) so a noisy
        # tiny detection can never win "largest so far".
        #
        # Right alongside it: the plate DETECTOR (YOLO, from
        # plate_detector.py) runs on this SAME per-frame bike crop, every
        # frame, for every live bike — just another YOLO model, no
        # different from the bike/mirror/indicator/orientation detectors
        # already running in this loop. Every detected plate box is mapped
        # straight to the bike it belongs to (bike.track_id).
        #
        # OCR (GOT-OCR 2.0 — actually READING the numbers off the plate) is
        # integrated INTO this same live loop, not deferred to the end of
        # the video. It's still called sparingly, not every frame:
        #   - only when THIS frame's plate box is a NEW BIGGEST (clearer)
        #     crop than any seen before for this track.
        # A smaller plate box than one already seen never re-triggers OCR.
        # UNLIKE before, this does NOT stop once a reading is found —
        # bike.plate_locked is no longer used to gate OCR calls, only as an
        # "ever read successfully" flag for logging/CSV. Every new-biggest
        # crop for a track gets its own OCR attempt and OVERWRITES
        # bike.plate_text if it returns non-empty text, on the theory that a
        # later, bigger/clearer crop is more trustworthy than an earlier,
        # smaller/blurrier one that may have OCR'd to garbage or a
        # hallucinated word. See Track.plate_text/plate_locked in track.py.
        for bike in live_tracks:
            try:
                bx1, by1, bx2, by2 = map(int, bike.to_tlbr())
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(width, bx2), min(height, by2)
                if bx2 > bx1 and by2 > by1:
                    bike_crop = frame[by1:by2, bx1:bx2]

                    with P.stage("logic.best_image"):
                        got_new_best = bike.update_best_image(
                            bike_crop,
                            min_area=MIN_BIKE_AREA_FOR_BEST_IMAGE,
                            full_frame=frame,
                            box=(bx1, by1, bx2, by2),
                        )
                    if got_new_best:
                        with P.stage("io.save_bike_snapshot"):     # writes JPEG (Drive!)
                            _save_bike_snapshot(bike, video_images_dir)

                    # ---- Plate DETECTOR only (use_ocr=False) here — OCR
                    # (if it runs at all this frame) is called separately,
                    # below, only on the winning box. offset=(bx1, by1) so
                    # every returned plate_box is already in FULL-FRAME
                    # coordinates (not just relative to bike_crop) — needed
                    # to draw it directly on `frame`.
                    try:
                        with P.stage("plate.detect"):
                            plate_dets = detect_plates_in_bike_crop(
                                bike_crop, plate_conf=PLATE_CONF, use_ocr=False,
                                offset=(bx1, by1), debug=False
                            )
                        P.metric("n_plate_dets", len(plate_dets))
                    except Exception:
                        log_exception(f"plate detector track_id={bike.track_id} at frame_idx={frame_idx}")
                        plate_dets = []

                    if plate_dets:
                        # YOLO found at least one plate box on this bike this
                        # frame -- mark it as "plate detected" for good,
                        # independent of the area/OCR gating below (that
                        # gating only decides whether THIS box is worth
                        # saving/OCR'ing, not whether a plate exists at all).
                        bike.plate_detected = True

                        # A bike normally has exactly one plate; if more than
                        # one box came back this frame (rare double-detection),
                        # keep only the largest one.
                        best_this_frame = max(
                            plate_dets,
                            key=lambda d: max(0, d["plate_box"][2] - d["plate_box"][0])
                                          * max(0, d["plate_box"][3] - d["plate_box"][1]),
                        )
                        px1, py1, px2, py2 = best_this_frame["plate_box"]
                        plate_area = max(0, px2 - px1) * max(0, py2 - py1)

                        is_new_biggest = (
                            plate_area >= MIN_PLATE_AREA_FOR_SAVE
                            and plate_area > best_plate_area_by_track.get(bike.track_id, 0)
                        )
                        if is_new_biggest:
                            best_plate_area_by_track[bike.track_id] = plate_area
                            with P.stage("io.save_plate_snapshot"):
                                _save_plate_snapshot(
                                    bike.track_id, best_this_frame["plate_crop"], video_images_dir
                                )

                            # ---- Live OCR — runs EVERY time this track gets
                            # a genuinely new-biggest plate crop (that's what
                            # is_new_biggest above already gates on), no
                            # matter how many times that happens over the
                            # track's life, and no matter whether an earlier,
                            # smaller/blurrier crop already produced some
                            # text. This intentionally does NOT stop after
                            # the first successful reading -- a small/blurry
                            # early crop can OCR to garbage (or a GOT-OCR
                            # hallucination like a random dictionary word),
                            # and the whole point of tracking best_plate_area
                            # per track is that a later, bigger/clearer crop
                            # should get its own OCR attempt and overwrite
                            # whatever the earlier read was. plate_crop from
                            # the detector is RGB (see
                            # detect_plates_in_bike_crop()); flip to BGR
                            # since read_plate_text_from_image() expects BGR
                            # (it converts back to RGB internally, matching
                            # every other entry point in plate_detector.py).
                            try:
                                plate_crop_bgr = cv2.cvtColor(
                                    best_this_frame["plate_crop"], cv2.COLOR_RGB2BGR
                                )
                                with P.stage("ocr.read"):
                                    ocr_text = read_plate_text_from_image(
                                        plate_crop_bgr, debug=False
                                    )
                                P.metric("n_ocr_calls", 1)
                            except Exception:
                                log_exception(
                                    f"live OCR track_id={bike.track_id} at frame_idx={frame_idx}"
                                )
                                ocr_text = ""

                            if ocr_text:
                                # bike.plate_locked is kept (set True here,
                                # same as before) purely as an "was this
                                # track EVER read successfully at least once"
                                # flag for finalize_track()'s log line / the
                                # CSV -- it is no longer used to block OCR
                                # from running again above.
                                bike.plate_text = ocr_text
                                bike.plate_locked = True
                                print(f"Track {bike.track_id}: plate OCR (live) -> "
                                      f"{ocr_text!r} on a new-biggest crop "
                                      f"(area={plate_area}px) -- will keep "
                                      f"re-OCR'ing this track if an even "
                                      f"bigger crop shows up later", flush=True)

                        # Draw EVERY plate box detected THIS frame on the
                        # annotated video, labeled with the OCR'd number
                        # once this track has one; falls back to a plain
                        # "Plate <id>" label until OCR resolves it.
                        for _pd in plate_dets:
                            try:
                                plate_label = bike.plate_text if bike.plate_text else f'Plate {bike.track_id}'
                                with P.stage("draw.boxes"):
                                    draw_box(frame, _pd["plate_box"], plate_label, (0, 255, 255))
                            except Exception:
                                log_exception(f"drawing plate box track_id={bike.track_id} at frame_idx={frame_idx}")
            except Exception:
                log_exception(f"update_best_image track_id={bike.track_id} at frame_idx={frame_idx}")

        print('================= Draw State ==============================')
        # No-op if TURN_ZONES is empty — safe to always call.
        with P.stage("draw.zones"):
            draw_zones(frame, TURN_ZONES)

        for bikes in live_tracks:
            try:
                bbox = bikes.to_tlbr()
                strs = "N"

                if bikes.is_turning and bikes.is_signal:
                    strs = "TO"
                if bikes.is_turning and not bikes.is_signal:
                    strs = "TF"

                x1, y1, x2, y2 = map(int, bbox)
                flow_dx, flow_dy, flow_mag, flow_angle = bike_flow_stats(flow, bbox)
                bikes.setFlow((flow_dx, flow_dy, flow_mag, flow_angle))

                with P.stage("draw.boxes"):
                    draw_box(frame, (x1, y1, x2, y2), f'bike: {bikes.track_id} - {strs}', bikes.box_color)
                for slot_idx, slot in enumerate(bikes.mirror_slots):
                    if slot["cord"] is None:
                        continue
                    with P.stage("draw.boxes"):
                        draw_box(frame, slot["cord"][1:], f'Mirror-{slot_idx} {bikes.track_id}', (0, 255, 0))
                for slot_idx, slot in enumerate(bikes.indicator_slots):
                    if slot["cord"] is None:
                        continue
                    label = f'Ind-{slot_idx} {bikes.track_id}' + (' ON' if slot["is_signal"] else '')
                    slot_color = (0, 255, 0) if slot["is_signal"] else (0, 165, 255)
                    with P.stage("draw.boxes"):
                        draw_box(frame, slot["cord"][1:], label, slot_color)
            except Exception:
                log_exception(f"drawing track_id={bikes.track_id} at frame_idx={frame_idx}")

        try:
            if writer:
                with P.stage("io.video_write"):        # mp4v encode + write
                    writer.write(frame)
        except Exception:
            log_exception(f"writer.write() at frame_idx={frame_idx}")
        #cv2.imshow("Video Tracking", frame)
        #if cv2.waitKey(1) & 0xFF == ord("q"):
        #    break

        print(f"[FRAME {frame_idx}] --- end ---", flush=True)
        P.end_frame()

    P.end_loop()
    with P.stage("post.release_capture_writer"):
        cap.release()
        if writer:
            writer.release()
    #cv2.destroyAllWindows()

    # Video ended while these tracks were still alive — DeepSORT never got
    # the chance to delete them naturally, so finalize them here instead.
    # Without this, any bike still on-screen in the last frame would simply
    # never get a CSV row.
    for t in list(known_tracks.values()):
        try:
            with P.stage("post.finalize_remaining"):
                finalize_track(t, records, video_images_dir)
        except Exception:
            log_exception(f"final finalize_track(track_id={t.track_id})")
    known_tracks.clear()

    # Re-encode the mp4v temp file to H.264 so it plays in browsers
    # (mp4v only plays in desktop players like VLC/WMP, not <video> tags).
    # -movflags +faststart moves metadata to the front of the file so it
    # can start streaming/playing before it's fully downloaded.
    _t_ff = time.perf_counter()
    if raw_output_path and os.path.exists(raw_output_path):
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", raw_output_path,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    output_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.remove(raw_output_path)
            print(f"Saved browser-compatible video: {output_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # ffmpeg missing or conversion failed -- keep the raw mp4v file
            # around (rename it back to the expected output_path) instead of
            # silently losing the processed video.
            print(f"WARNING: ffmpeg re-encode failed ({e}); "
                  f"keeping original mp4v file (plays in VLC/media players, not browsers).")
            os.replace(raw_output_path, output_path)
    P.record("post.ffmpeg_h264_reencode", (time.perf_counter() - _t_ff) * 1000.0)

    # ------------------------------------------------------------------
    # Free every detector/encoder model from CPU/GPU memory now that the
    # frame loop is done — just to release CPU/GPU memory promptly, not to
    # protect OCR from a conflict (GOT-OCR 2.0 is PyTorch-based, so it
    # already shared the CUDA context fine with these models while OCR ran
    # live, inside the loop above — see the plate DETECTOR + live-OCR block).
    # ------------------------------------------------------------------
    _t_free = time.perf_counter()
    try:
        del yolo, mirror_detector, indicator_detector, orientation_detector, encoder
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            log_exception("torch.cuda.empty_cache()/ipc_collect() after video loop")
    gc.collect()
    P.record("post.free_models_gc", (time.perf_counter() - _t_free) * 1000.0)
    print(f"[{vidName}] All detector models unloaded, CPU/GPU freed "
          f"(cuda_available={torch.cuda.is_available()}).", flush=True)

    # ------------------------------------------------------------------
    # FALLBACK OCR pass — most bikes already have their plate_number filled
    # in by finalize_track() (from live, in-loop OCR — see the plate
    # DETECTOR + live-OCR block in the main loop above). This pass only
    # picks up any track that reached the end of the video WITHOUT ever
    # getting a successful reading (rec["plate_number"] still empty), and
    # gives it exactly one more try, reading straight off its saved
    # images_dir/<track_id>/plate_crop.jpg. No detection happens here.
    # ------------------------------------------------------------------
    unresolved = [rec for rec in records if not rec.get("plate_number")]
    if unresolved:
        try:
            warm_up_ocr()
        except Exception:
            log_exception("warm_up_ocr() in post-loop fallback OCR pass")

        for rec in unresolved:
            plate_text = ""
            bike_dir = os.path.join(video_images_dir, str(rec["track_id"]))
            plate_crop_path = os.path.join(bike_dir, "plate_crop.jpg")

            if os.path.exists(plate_crop_path):
                try:
                    with P.stage("post.fallback_ocr"):
                        plate_text = read_plate_text_from_image(plate_crop_path, debug=False)
                except Exception:
                    log_exception(f"fallback OCR on track_id={rec['track_id']} ({plate_crop_path})")
            else:
                print(f"Track {rec['track_id']}: no plate_crop.jpg was ever produced during the "
                      f"frame loop — treating plate as unreadable.", flush=True)

            rec["plate_number"] = plate_text
            print(f"Track {rec['track_id']}: fallback OCR -> plate={plate_text!r}", flush=True)

    # ------------------------------------------------------------------
    # Write the summary CSV exactly once, now that every bike's
    # plate_number is known (or confirmed unreadable). csv.writer is used
    # so a plate string (or anything else) containing a comma can never
    # corrupt the file the way the old manual ",".join(...) writing could.
    # ------------------------------------------------------------------
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "bike_id", "plate_number", "plate_detected", "has_both_mirrors",
            "has_both_indicators", "took_turn", "indicator_on_while_turning",
        ])
        for rec in records:
            csv_writer.writerow([
                rec["track_id"],
                rec.get("plate_number", ""),
                rec.get("plate_detected", False),
                rec["mirror_seen_both"],
                rec["indicator_seen_both"],
                rec["ever_turned"],
                rec["ever_signaled_while_turning"],
            ])
    print(f"[{vidName}] Summary CSV written -> {summary_csv_path} ({len(records)} bike(s)).", flush=True)

    P.end_run(extra={"bikes_in_csv": len(records)})

    return {
        "output_video_path": output_path,
        "summary_csv_path": summary_csv_path,
        "video_images_dir": video_images_dir,
        "video_dir": video_dir,
        "video_base": video_base,
    }


if __name__ == "__main__":

    # python deepsortVideo.py --compare run_T4.json run_L4.json   (ya bina args ke: latest 4 JSONs)
    if len(sys.argv) >= 2 and sys.argv[1] == "--compare":
        compare_reports(sys.argv[2:])
        sys.exit(0)

    for vid in os.listdir(VIDEO_PATH):
        print(f"===== Starting video: {vid} =====", flush=True)
        try:
            result = run_video(os.path.join(VIDEO_PATH, vid), OUTPUT_PATH, display_name=vid)
            print(f"Result: {result}", flush=True)
        except Exception:
            log_exception(f"run_video() top-level for video={vid}")
        print(f"===== Finished (or failed) video: {vid} =====", flush=True)

