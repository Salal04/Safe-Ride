"""Backward-compatible public interface for plate detection and OCR."""
from config import *  # Preserve configuration names exposed by the old module.
from . import plate_detection as _detection
from . import plate_ocr as _ocr_module
from .plate_detection import (
    detect_plates_in_bike_crop, detect_plates_in_frame, detect_plates_in_video,
    get_plate_from_bike_image, get_plate_from_precropped_bike,
    warm_up_plate_detector,
)
from .plate_ocr import read_plate_text_from_image, warm_up_ocr

_PRIVATE_DETECTION_NAMES = {
    "_bike_detector", "_plate_detector", "_save_received_image",
    "_get_bike_detector", "_get_plate_detector",
}
_PRIVATE_OCR_NAMES = {
    "_ocr", "_log_plate_error", "_Watchdog", "_in_colab",
    "_describe_image", "_debug_show", "_get_ocr", "_run_ocr",
}


def __getattr__(name):
    if name in _PRIVATE_DETECTION_NAMES:
        return getattr(_detection, name)
    if name in _PRIVATE_OCR_NAMES:
        return getattr(_ocr_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
