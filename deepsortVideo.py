from __future__ import annotations

import logging
import os
import sys
import traceback

import torch

from config import *

# Keep the existing log target and startup messages used by the CLI and API.
_LOG_DIR_FOR_LOGGING = str(LOG_DIR)
os.makedirs(_LOG_DIR_FOR_LOGGING, exist_ok=True)
logger = logging.getLogger("deepsortVideo")
logger.setLevel(getattr(logging, LOG_LEVEL))
logger.handlers.clear()
_file_handler = logging.FileHandler(
    os.path.join(_LOG_DIR_FOR_LOGGING, "run_debug.log"), mode="a", encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_file_handler)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_console_handler)


def log_exception(context_msg):
    """Write the current exception and its traceback to the shared log."""
    logger.error("❌ EXCEPTION in %s", context_msg)
    logger.error(traceback.format_exc())
    for handler in logger.handlers:
        handler.flush()


# Preserve the original PyTorch checkpoint compatibility behavior.
_orig_load = torch.load
torch.load = lambda *args, **kwargs: _orig_load(
    *args, **{**kwargs, "weights_only": False}
)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
device = ("cuda" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else DEVICE

from backend.vision_geometry import (
    bike_flow_stats, build_zone_mask, compute_iou, draw_zones, is_box_inside,
    xyxy_to_tlwh, zone_overlap_ratio,
)
from backend.pipeline_profiler import P, PipelineProfiler, compare_reports, render_report, _safe_slug
from backend.pipeline_models import (
    blink_detector, build_osnet_encoder,
    extract_indicator_crop, predict_from_track,
)
from backend.pipeline_tracking import (
    _save_bike_snapshot, _save_plate_snapshot, active_tracks, best_match_track,
    draw_box, finalize_track, getDetections,
)
from backend import pipeline_runner


def run_video(video_path, output_path=None, display_name=None):
    """Run inference, forwarding the module-level turn zones for compatibility."""
    return pipeline_runner.run_video(
        video_path, output_path, display_name, turn_zones=TURN_ZONES
    )


if __name__ == "__main__":
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
