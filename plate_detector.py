"""
plate_detector.py — license plate detector + OCR module (with debug mode).

All functions auto-load bike_detector, plate_detector, and OCR
internally (set your model paths below, once).

DEBUG MODE
----------
Every public function now takes a `debug=False` kwarg. Turn it on (debug=True)
to get, at each stage of the pipeline:
  - a print of the image's type / shape / dtype right when it's RECEIVED
  - a print right before it goes into a model for PREDICTION
  - the image itself displayed inline (Colab-safe: uses cv2_imshow when
    running in Colab, falls back to matplotlib everywhere else)
  - explicit messages when detection finds nothing, or OCR finds no
    text / everything is below the confidence threshold
This is meant to be turned off (debug=False, the default) for normal/video
runs since it will spam your notebook with images otherwise.

DETECTION vs. OCR — split on purpose
-------------------------------------
For a live per-frame pipeline (see deepsortVideo.py), the plate DETECTOR
(YOLO — just locating and cropping the plate) runs every frame, right
alongside your other per-frame YOLO models. OCR (GOT-OCR 2.0 — reading the
actual characters) is meant to be called separately, on an already-cropped
plate image, only when the caller decides it's worth paying for (e.g. a
new largest/clearest plate crop for a track that hasn't been read yet —
see Track.plate_locked / deepsortVideo.py's main loop). Two functions make
this split easy:
  - detect_plates_in_bike_crop(..., use_ocr=False)  -> detection ONLY
  - read_plate_text_from_image(...)                 -> OCR ONLY, on an
    already-cropped plate image (path or array)
warm_up_plate_detector() / warm_up_ocr() preload each side ahead of time.

OCR ENGINE: GOT-OCR 2.0 (stepfun-ai/GOT-OCR2_0, via transformers).
EasyOCR was dropped — on real plate crops (small, blurry, low-res) it kept
missing text at the DETECTION stage (CRAFT's default thresholds are
conservative, so borderline/small text regions never even become a box,
regardless of param tuning). RapidOCR (PP-OCR/ONNX) was used next and
matched PaddleOCR's accuracy without the paddlepaddle install/CUDA-context
headaches, but GOT-OCR 2.0 (a ~580M param vision-language transformer)
gives noticeably better accuracy on small/blurry/low-res crops like plate
photos, at the cost of being heavier and needing a GPU for reasonable
speed.

GOT-OCR 2.0 is a PyTorch/transformers model, so it shares this process's
existing CUDA context with the PyTorch/Ultralytics (YOLO) models already
loaded — same situation EasyOCR was in, no separate onnxruntime session
like RapidOCR needed.

Unlike RapidOCR/EasyOCR, GOT-OCR 2.0 does NOT return a per-line confidence
score — model.chat() just returns one recognized text string for the whole
crop. So ocr_conf_thresh (still accepted by every function below, for
backwards compatibility with the rest of this module / deepsortVideo.py)
no longer filters anything by confidence; a non-empty, allowlist-cleaned
result is simply accepted. See _run_ocr() below.

Install (done manually in the Colab notebook, not by this file):
    pip install torch torchvision transformers accelerate --break-system-packages
    pip install tiktoken verovio --break-system-packages
"""

import os
import cv2
import traceback
import threading
import numpy as np
from config import *
from datetime import datetime
from collections import Counter


def _log_plate_error(context_msg):
    print(f"❌ EXCEPTION in plate_detector.py ({context_msg})", flush=True)
    traceback.print_exc()


class _Watchdog:
    """Prints a warning if a `with`-block takes longer than `timeout`
    seconds, WITHOUT killing it — just so we can tell "it's hung" apart
    from "it silently crashed" in the logs. Doesn't fire if the block
    finishes in time."""
    def __init__(self, label, timeout=20):
        self.label = label
        self.timeout = timeout
        self._timer = None

    def __enter__(self):
        self._timer = threading.Timer(self.timeout, self._fire)
        self._timer.daemon = True
        self._timer.start()
        return self

    def _fire(self):
        print(f"⚠️ WATCHDOG: '{self.label}' has been running for over "
              f"{self.timeout}s and hasn't returned yet — this looks like "
              f"a HANG (e.g. stuck network download, or a CUDA/GPU context "
              f"conflict), not a fast crash.", flush=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._timer:
            self._timer.cancel()
        return False

# ============================================================
# MODEL PATHS — edit these once
# ============================================================

_bike_detector = None
_plate_detector = None
_ocr = None


# ============================================================
# DEBUG HELPERS
# ============================================================
def _in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _describe_image(img, label):
    """Print type/shape/dtype info about whatever we were just handed."""
    if img is None:
        print(f"[DEBUG] {label}: received None")
        return
    if isinstance(img, np.ndarray):
        print(f"[DEBUG] {label}: numpy.ndarray | shape={img.shape} | "
              f"dtype={img.dtype} | min={img.min()} max={img.max()}")
    else:
        print(f"[DEBUG] {label}: type={type(img)} (NOT a numpy array — "
              f"this is probably why something downstream is failing)")


def _debug_show(img, title, debug):
    """Show an image inline, Colab-safe, only if debug=True."""
    if not debug:
        return
    if img is None or (isinstance(img, np.ndarray) and img.size == 0):
        print(f"[DEBUG] {title}: nothing to show (image is None/empty)")
        return

    print(f"[DEBUG] showing: {title}")
    try:
        if _in_colab():
            from google.colab.patches import cv2_imshow
            cv2_imshow(img)  # expects BGR, like normal cv2 images
        else:
            import matplotlib.pyplot as plt
            # matplotlib expects RGB; assume BGR in (our normal convention)
            # unless the caller tells us otherwise via title hint.
            if img.ndim == 3 and img.shape[2] == 3:
                display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                display_img = img
            plt.figure()
            plt.title(title)
            plt.axis("off")
            plt.imshow(display_img)
            plt.show()
    except Exception as e:
        print(f"[DEBUG] could not display '{title}': {e}")


def _save_received_image(img, save_dir=IMAGE_SAVE_DIR, original_path=None):
    """Incoming image ko save_dir folder mein save karta hai.

    Agar original_path (file path) mojood hai to usi ka filename use hota
    hai, warna timestamp se unique filename bana diya jata hai.
    """
    if img is None or (isinstance(img, np.ndarray) and img.size == 0):
        return None

    os.makedirs(save_dir, exist_ok=True)

    if original_path:
        filename = os.path.basename(original_path)
    else:
        filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"

    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, img)
    print(f"📁 Image saved to: {save_path}")
    return save_path


def _get_bike_detector():
    global _bike_detector
    if _bike_detector is None:
        print("[plate_detector] Loading bike_detector YOLO model (first call, lazy-load)...", flush=True)
        try:
            from ultralytics import YOLO
            _bike_detector = YOLO(BIKE_MODEL_PATH)
            print("[plate_detector] bike_detector loaded OK.", flush=True)
        except Exception:
            _log_plate_error("_get_bike_detector() model load")
            raise
    return _bike_detector


def _get_plate_detector():
    global _plate_detector
    if _plate_detector is None:
        print("[plate_detector] Loading plate_detector YOLO model (first call, lazy-load)...", flush=True)
        try:
            from ultralytics import YOLO
            _plate_detector = YOLO(PLATE_MODEL_PATH)
            print("[plate_detector] plate_detector loaded OK.", flush=True)
            try:
                print(f"[plate_detector] plate_detector class names (id -> label): "
                      f"{_plate_detector.names} -- if PLATE_CLASS_ID above doesn't match "
                      f"the 'plate'/'number-plate' entry here, plate detection can return "
                      f"boxes for the WRONG class (e.g. the whole motorcycle) and "
                      f"PLATE_CLASS_ID should be updated to match.", flush=True)
            except Exception:
                pass
        except Exception:
            _log_plate_error("_get_plate_detector() model load")
            raise
    return _plate_detector


def _get_ocr():
    """Lazy-loads GOT-OCR 2.0 (model + tokenizer). Returns (model, tokenizer).

    NOTE: this is a PyTorch/transformers model (~580M params, ~1.5-2GB
    download on first call) — much heavier than RapidOCR was. It shares
    this process's existing CUDA context with the YOLO models, same as
    EasyOCR did, so it competes for the same GPU memory. If you're tight on
    VRAM alongside bike/plate YOLO models, watch for CUDA OOM here.
    """
    global _ocr
    if _ocr is None:
        print("[plate_detector] Loading GOT-OCR 2.0 (first call, lazy-load, "
              "may download ~1.5-2GB of weights)...", flush=True)
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            with _Watchdog("GOT-OCR 2.0 init", timeout=180):
                tokenizer = AutoTokenizer.from_pretrained(
                    GOT_OCR_MODEL_ID, trust_remote_code=True
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = AutoModel.from_pretrained(
                    GOT_OCR_MODEL_ID,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map=device,
                    use_safetensors=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                model = model.eval().to(device)

            _ocr = (model, tokenizer)
            print(f"[plate_detector] GOT-OCR 2.0 loaded OK on device={device}.", flush=True)
        except Exception:
            _log_plate_error("_get_ocr() GOT-OCR 2.0 load")
            raise
    return _ocr


def warm_up_ocr():
    """Call this ONCE, ideally before the main video-processing loop starts
    (see deepsortVideo.py), so GOT-OCR 2.0's one-time model download/init
    happens up front — while nothing else is competing for CPU/network/GPU
    memory — instead of happening in the middle of frame processing."""
    print("[plate_detector] Warming up GOT-OCR 2.0 before video processing starts...", flush=True)
    _get_ocr()
    print("[plate_detector] GOT-OCR 2.0 warm-up done.", flush=True)


def warm_up_plate_detector():
    """Call this ONCE, before the main per-frame loop starts (see
    deepsortVideo.py) — right alongside the bike/mirror/indicator/
    orientation YOLO models — so the plate DETECTOR's one-time weight load
    happens up front instead of silently happening on frame 1.

    This is just another YOLO model (like the bike/mirror/indicator/
    orientation detectors), so it's fine to run it every frame inside the
    main loop. NOTE: unlike RapidOCR, GOT-OCR 2.0 IS a PyTorch/transformers
    model and DOES share the same CUDA context/GPU memory as these YOLO
    models -- if you hit CUDA OOM, that's why (see _get_ocr() above).
    """
    print("[plate_detector] Warming up plate detector (YOLO) before video processing starts...", flush=True)
    _get_plate_detector()
    print("[plate_detector] Plate detector warm-up done.", flush=True)


def _run_ocr(plate_crop, ocr_conf_thresh, debug=False):
    """Run OCR on a plate crop (RGB numpy array), return recognized text or "".

    NOTE: ocr_conf_thresh is accepted for backwards compatibility with every
    caller in this module / deepsortVideo.py, but GOT-OCR 2.0's model.chat()
    does not return a per-line confidence score the way RapidOCR did -- so
    it is NOT used to filter/reject text here. The only filtering that still
    happens is the OCR_ALLOWLIST character cleanup below.
    """
    if debug:
        _describe_image(plate_crop, "OCR input (plate_crop) RECEIVED")

    if plate_crop is None or plate_crop.size == 0:
        if debug:
            print("[DEBUG] OCR skipped: plate_crop is None/empty")
        return ""

    resized = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    if debug:
        _describe_image(resized, "OCR input AFTER resize (going into GOT-OCR 2.0)")
        _debug_show(resized, "OCR input crop", debug)

    # GOT-OCR 2.0's chat() takes a file path, not an array -- write the crop
    # out to a scratch file every call. plate_crop/resized is RGB (this
    # module's convention everywhere) so flip back to BGR for cv2.imwrite.
    try:
        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(GOT_OCR_TMP_PATH, resized_bgr)
    except Exception:
        _log_plate_error("_run_ocr(): failed writing temp crop for GOT-OCR 2.0")
        return ""

    print("[plate_detector] calling GOT-OCR 2.0 on resized crop...", flush=True)
    try:
        model, tokenizer = _get_ocr()
        with _Watchdog("GOT-OCR 2.0 inference", timeout=60):
            raw_text = model.chat(tokenizer, GOT_OCR_TMP_PATH, ocr_type=GOT_OCR_TYPE)
    except Exception:
        _log_plate_error("_run_ocr(): GOT-OCR 2.0 model.chat() call failed")
        return ""
    print("[plate_detector] GOT-OCR 2.0 call finished.", flush=True)

    if debug:
        print(f"[DEBUG] raw OCR result: {raw_text!r}")

    if not raw_text or not raw_text.strip():
        if debug:
            print("[DEBUG] OCR FAILED: GOT-OCR 2.0 returned no text at all "
                  "(likely blurry/too small/no text in crop)")
        return ""

    # Plates can OCR as more than one line (e.g. city name on top, number
    # below) -- GOT-OCR 2.0 may separate these with newlines, so join
    # everything together the same way the old multi-line accumulation did.
    cleaned = raw_text.upper().replace(" ", "").replace("\n", "")
    if OCR_ALLOWLIST is not None:
        cleaned = "".join(ch for ch in cleaned if ch in OCR_ALLOWLIST)

    if not cleaned:
        if debug:
            print(f"[DEBUG] OCR FAILED: raw text {raw_text!r} had nothing left "
                  f"after OCR_ALLOWLIST cleanup.")
        return ""

    if debug:
        print(f"[DEBUG] OCR candidate ACCEPTED: raw={raw_text!r} -> cleaned={cleaned!r}")

    return cleaned


def read_plate_text_from_image(plate_crop_bgr, ocr_conf_thresh=OCR_CONF, debug=False):
    """OCR-ONLY entry point — no detection model runs here at all.

    Use this on a plate image that was ALREADY located and cropped by the
    plate DETECTOR earlier (e.g. saved to disk as plate_crop.jpg during a
    per-frame loop — see detect_plates_in_bike_crop(..., use_ocr=False) and
    deepsortVideo.py's main loop) and is now just being re-loaded for the
    one-time, end-of-video OCR pass.

    Parameters
    ----------
    plate_crop_bgr : np.ndarray or str
        Either an already-loaded plate crop in standard OpenCV BGR order
        (e.g. straight out of cv2.imread()), or a file path to one — a
        path is loaded with cv2.imread() automatically.
    debug : bool
        If True, prints/shows OCR-stage debug info (same as the other
        functions in this module).

    Returns
    -------
    str : recognized plate text, or "" if nothing readable was found.
    """
    if isinstance(plate_crop_bgr, str):
        plate_crop_bgr = cv2.imread(plate_crop_bgr)

    if plate_crop_bgr is None or plate_crop_bgr.size == 0:
        if debug:
            print("[DEBUG] read_plate_text_from_image: empty/None input")
        return ""

    # _run_ocr() expects the same RGB convention detect_plates_in_bike_crop()
    # always fed it before (that function converts BGR->RGB before slicing
    # the plate crop out) — flip it here too, so OCR sees the exact same
    # pixel order it always has, whether the crop just came straight out of
    # the detector or was saved to disk and re-loaded via cv2.imread (BGR).
    plate_crop_rgb = cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2RGB)

    try:
        return _run_ocr(plate_crop_rgb, ocr_conf_thresh, debug=debug)
    except Exception:
        _log_plate_error("read_plate_text_from_image(): _run_ocr() failed")
        return ""


def detect_plates_in_bike_crop(bike_crop, plate_conf=PLATE_CONF, ocr_conf_thresh=OCR_CONF,
                                use_ocr=True, offset=(0, 0), debug=False):
    """Detect plate(s) in an already-cropped bike image, with OCR."""
    detections = []

    if debug:
        _describe_image(bike_crop, "detect_plates_in_bike_crop: bike_crop RECEIVED")
        _debug_show(bike_crop, "bike_crop (as received)", debug)

    if bike_crop is None or bike_crop.size == 0:
        if debug:
            print("[DEBUG] detect_plates_in_bike_crop: empty/None input, returning []")
        return detections

    # Used below (PLATE_MAX_AREA_RATIO) to reject a "plate" box that's
    # actually most of the bike -- see the loop below.
    bike_h, bike_w = bike_crop.shape[:2]
    bike_area = bike_h * bike_w

    bike_crop = cv2.cvtColor(bike_crop, cv2.COLOR_BGR2RGB)  # plate model expects RGB, frame is BGR (OpenCV)

    if debug:
        _describe_image(bike_crop, "bike_crop AFTER BGR->RGB, going into plate model")

    ox, oy = offset
    # Filter to ONLY the plate class when we know its id -- otherwise this
    # call returns every class the model knows about (e.g. "motorcycle"),
    # any of which could get treated as if it were a plate. See the
    # PLATE_CLASS_ID comment near the top of this file.
    predict_kwargs = dict(conf=plate_conf, verbose=False)
    if PLATE_CLASS_ID is not None:
        predict_kwargs["classes"] = [PLATE_CLASS_ID]

    print("[plate_detector] calling plate_detector.predict()...", flush=True)
    try:
        plate_results = _get_plate_detector().predict(bike_crop, **predict_kwargs)
    except Exception:
        _log_plate_error("detect_plates_in_bike_crop(): plate_detector.predict() failed")
        return detections
    print("[plate_detector] plate_detector.predict() finished.", flush=True)

    n_boxes = sum(len(pr.boxes) for pr in plate_results)
    if debug:
        print(f"[DEBUG] plate detector found {n_boxes} box(es) at conf>={plate_conf}")
        if n_boxes == 0:
            print("[DEBUG] DETECTION FAILED: no plate boxes found in this bike_crop. "
                  "Try lowering plate_conf, or check PLATE_MODEL_PATH / classes.")

    for pr in plate_results:
        for plate_box in pr.boxes:
            px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
            px1, py1 = max(0, px1), max(0, py1)

            # Safety net, independent of PLATE_CLASS_ID: a real plate is a
            # small fraction of the bike it's on. A box covering most of
            # the bike_crop is almost certainly the wrong class (or a bad
            # detection) -- exactly what would otherwise let the whole
            # bike through disguised as a "plate" crop.
            box_area = max(0, px2 - px1) * max(0, py2 - py1)
            if bike_area > 0 and box_area >= PLATE_MAX_AREA_RATIO * bike_area:
                if debug:
                    print(f"[DEBUG] REJECTING 'plate' box {(px1, py1, px2, py2)} -> covers "
                          f"{box_area / bike_area:.0%} of the bike crop (>= "
                          f"{PLATE_MAX_AREA_RATIO:.0%} threshold) -- almost certainly NOT a "
                          f"real plate. Check PLATE_CLASS_ID against this model's class "
                          f"names (printed when it loaded).")
                continue

            plate_crop = bike_crop[py1:py2, px1:px2]
            if plate_crop.size == 0:
                if debug:
                    print(f"[DEBUG] skipping a plate box {(px1, py1, px2, py2)} "
                          f"-> crop came out empty")
                continue

            if debug:
                _debug_show(plate_crop, f"detected plate crop @ {(px1, py1, px2, py2)}", debug)

            plate_text = _run_ocr(plate_crop, ocr_conf_thresh, debug=debug) if use_ocr else ""

            if debug:
                print(f"[DEBUG] plate box {(px1, py1, px2, py2)} -> "
                      f"plate_text={plate_text!r}")

            detections.append({
                "plate_box": (ox + px1, oy + py1, ox + px2, oy + py2),
                "plate_crop": plate_crop,
                "plate_text": plate_text,
            })

    return detections


def detect_plates_in_frame(frame, bike_conf=BIKE_CONF, plate_conf=PLATE_CONF,
                            ocr_conf_thresh=OCR_CONF, use_ocr=True, debug=False):
    """Detect bike(s), then plate(s) inside each bike, in a full frame, with OCR."""
    detections = []

    if debug:
        _describe_image(frame, "detect_plates_in_frame: frame RECEIVED")
        _debug_show(frame, "full frame (as received)", debug)

    bike_results = _get_bike_detector().predict(frame, conf=bike_conf,
                                                 classes=[BIKE_CLASS_ID], verbose=False)

    n_bikes = sum(len(br.boxes) for br in bike_results)
    if debug:
        print(f"[DEBUG] bike detector found {n_bikes} bike box(es) at conf>={bike_conf}")
        if n_bikes == 0:
            print("[DEBUG] DETECTION FAILED: no bikes found in this frame. "
                  "Check bike_conf, BIKE_CLASS_ID, or BIKE_MODEL_PATH.")

    for br in bike_results:
        for bike_box in br.boxes:
            bx1, by1, bx2, by2 = map(int, bike_box.xyxy[0])
            bx1, by1 = max(0, bx1), max(0, by1)
            bike_roi = frame[by1:by2, bx1:bx2]

            if bike_roi.size == 0:
                if debug:
                    print(f"[DEBUG] skipping bike box {(bx1, by1, bx2, by2)} "
                          f"-> crop came out empty")
                continue

            if debug:
                _debug_show(bike_roi, f"bike crop @ {(bx1, by1, bx2, by2)} "
                                       f"(going into plate detection)", debug)

            plate_dets = detect_plates_in_bike_crop(
                bike_roi, plate_conf=plate_conf, ocr_conf_thresh=ocr_conf_thresh,
                use_ocr=use_ocr, offset=(bx1, by1), debug=debug
            )

            for det in plate_dets:
                det["bike_box"] = (bx1, by1, bx2, by2)
                detections.append(det)

    return detections


def detect_plates_in_video(input_path, output_path, bike_conf=BIKE_CONF, plate_conf=PLATE_CONF,
                            ocr_conf_thresh=OCR_CONF, use_ocr=True, draw=True,
                            progress_every=PLATE_VIDEO_PROGRESS_EVERY, debug=False, debug_every=PLATE_VIDEO_DEBUG_EVERY):
    """Run plate detection + OCR on a full video, save annotated output, return final plate.

    debug=True will print/show per-frame debug info. Since that's a LOT of
    output for a video, use `debug_every` to only debug every Nth frame
    (e.g. debug_every=30 -> debug once per ~1s of a 30fps video).
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if debug:
        print(f"[DEBUG] video opened: {input_path} | {width}x{height} @ {fps}fps | "
              f"{total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_detections = []
    all_texts = []
    last_known_text = ""
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_debug = debug and (frame_idx % debug_every == 0)
        if frame_debug:
            print(f"\n[DEBUG] ===== frame {frame_idx} =====")

        dets = detect_plates_in_frame(
            frame, bike_conf=bike_conf, plate_conf=plate_conf,
            ocr_conf_thresh=ocr_conf_thresh, use_ocr=use_ocr, debug=frame_debug
        )

        for det in dets:
            if det["plate_text"]:
                all_texts.append(det["plate_text"])
                last_known_text = det["plate_text"]

            if draw:
                bx1, by1, bx2, by2 = det["bike_box"]
                px1, py1, px2, py2 = det["plate_box"]

                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 3)

                label = det["plate_text"] if det["plate_text"] else last_known_text
                if label:
                    (text_w, text_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
                    )
                    cv2.rectangle(frame, (px1, py1 - text_h - 15),
                                  (px1 + text_w + 10, py1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (px1 + 5, py1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        out.write(frame)
        all_detections.append({"frame_idx": frame_idx, "detections": dets})

        frame_idx += 1
        if progress_every and frame_idx % progress_every == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"\n✅ Output video saved to: {output_path}")

    final_plate, final_plate_count = None, 0
    if all_texts:
        final_plate, final_plate_count = Counter(all_texts).most_common(1)[0]
        print(f"✅ Final Plate Number: {final_plate} (seen {final_plate_count} times)")
    elif use_ocr:
        print("❌ Koi plate clearly nahi parhi ja saki")

    return {
        "output_path": output_path,
        "final_plate": final_plate,
        "final_plate_count": final_plate_count,
        "all_detections": all_detections,
    }


def get_plate_from_precropped_bike(bike_crop, plate_conf=PLATE_CONF, ocr_conf_thresh=OCR_CONF,
                                    use_ocr=True, debug=True,
                                    save_image=True, save_dir=IMAGE_SAVE_DIR):
    """Given an image that is ALREADY a tight crop of one bike (e.g. the
    largest crop a tracker collected for that bike over its lifetime), run
    the plate detector + OCR directly on it — skipping the bike-detection
    step entirely, since the crop was already located and sized by the
    tracker.

    This is the intended entry point for a tracking pipeline that keeps one
    "best" (largest) image per tracked bike and only wants to pay for plate
    OCR once, at the end of that bike's track, rather than every frame.

    Parameters
    ----------
    bike_crop : np.ndarray
        BGR image, already cropped to roughly one bike.
    debug : bool
        If True, prints image type/shape at each stage and shows the
        images inline (Colab-safe).
    save_image : bool
        If True (default), the received bike_crop is saved into save_dir.
    save_dir : str
        Folder to save the received image into.

    Returns
    -------
    dict with:
        "plate_text"  : best (first non-empty) plate text found, or None
        "detections"  : full list of plate detections (box/crop/text each)
    """
    if debug:
        _describe_image(bike_crop, "get_plate_from_precropped_bike: bike_crop RECEIVED")
        _debug_show(bike_crop, "input bike_crop", debug)

    if bike_crop is None or bike_crop.size == 0:
        if debug:
            print("[DEBUG] get_plate_from_precropped_bike: empty/None input")
        return {"plate_text": None, "detections": []}

    if save_image:
        _save_received_image(bike_crop, save_dir=save_dir)

    bike_crop = cv2.cvtColor(bike_crop, cv2.COLOR_BGR2RGB)

    if debug:
        _describe_image(bike_crop, "bike_crop AFTER BGR->RGB, going into plate detection")

    print("[plate_detector] get_plate_from_precropped_bike: entering detect_plates_in_bike_crop()...", flush=True)
    try:
        dets = detect_plates_in_bike_crop(
            bike_crop, plate_conf=plate_conf, ocr_conf_thresh=ocr_conf_thresh,
            use_ocr=use_ocr, debug=debug
        )
    except Exception:
        _log_plate_error("get_plate_from_precropped_bike(): detect_plates_in_bike_crop() failed")
        return {"plate_text": None, "detections": []}
    print("[plate_detector] get_plate_from_precropped_bike: detect_plates_in_bike_crop() returned.", flush=True)

    plate_text = next((d["plate_text"] for d in dets if d["plate_text"]), None)

    if debug:
        if plate_text is None:
            print("[DEBUG] get_plate_from_precropped_bike: FINAL RESULT = None "
                  "(no plate detected, or OCR failed on all detected plates)")
        else:
            print(f"[DEBUG] get_plate_from_precropped_bike: FINAL RESULT = {plate_text!r}")

    return {
        "plate_text": plate_text,
        "detections": dets,
    }


def get_plate_from_bike_image(image, bike_conf=BIKE_CONF, plate_conf=PLATE_CONF,
                               ocr_conf_thresh=OCR_CONF, use_ocr=True, debug=True,
                               save_image=True, save_dir=IMAGE_SAVE_DIR):
    """Give a bike image (path or array), get back the detected plate text.

    save_image=True (default) -> incoming image save_dir folder mein
    automatically save ho jati hai. save_image=False karke isko band
    kiya ja sakta hai.
    """
    if debug:
        print(f"[DEBUG] get_plate_from_bike_image: input arg type={type(image)}")

    if isinstance(image, str):
        frame = cv2.imread(image)
        if frame is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image}")
        if debug:
            print(f"[DEBUG] loaded image from path '{image}'")
        original_path = image
    else:
        frame = image
        original_path = None

    if save_image:
        _save_received_image(frame, save_dir=save_dir, original_path=original_path)

    if debug:
        _describe_image(frame, "get_plate_from_bike_image: frame RECEIVED")
        _debug_show(frame, "input frame", debug)

    dets = detect_plates_in_frame(
        frame, bike_conf=bike_conf, plate_conf=plate_conf,
        ocr_conf_thresh=ocr_conf_thresh, use_ocr=use_ocr, debug=debug
    )

    plate_text = next((d["plate_text"] for d in dets if d["plate_text"]), None)

    if debug:
        if plate_text is None:
            print("[DEBUG] get_plate_from_bike_image: FINAL RESULT = None "
                  "(no bike/plate detected, or OCR failed)")
        else:
            print(f"[DEBUG] get_plate_from_bike_image: FINAL RESULT = {plate_text!r}")

    return {
        "plate_text": plate_text,
        "detections": dets,
    }
