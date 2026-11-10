# Backend package

`backend_server.py` and `deepsortVideo.py` remain at the project root as stable
launch/import entry points. Backend implementation and model code live here.

- `backend_support.py`: API conversion, CSV, manifest, and response helpers.
- `pipeline_runner.py`: video inference orchestration.
- `pipeline_models.py`, `blink_model.py`: blink and OSNet adapters/models.
- `pipeline_tracking.py`: detections, track matching, and bike artifacts.
- `plate_detector.py`: compatibility facade for plate functions.
- `plate_detection.py`, `plate_ocr.py`: detection workflows and OCR engine.
- `vision_geometry.py`: box geometry and junction zone masks.
- `pipeline_profiler.py`, `profiler_reports.py`: profiling and report output.

`config.py` remains at the root because existing scripts import it directly.
`tools/` contains manually runnable legacy utilities; root launch shims keep
previous script names available.
