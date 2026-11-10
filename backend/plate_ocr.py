"""GOT-OCR initialization and plate text recognition."""
import threading
import traceback

import cv2
import numpy as np

from config import *

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

_ocr = None

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
