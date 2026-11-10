"""Human-readable profiling reports and cross-run comparisons."""
import datetime
import glob
import json
import os

from config import LOG_DIR, PROFILE_SAVE_DIR_NAME


def _bar(frac, width=22):
    n = max(0, min(width, int(round(frac * width))))
    return "#" * n + "." * (width - n)


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
    L.append(" per-stage mean ms/frame   (Nx = baseline / this GPU, >1 matlab tez)")
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



