# SafeRide Vision — Review Questions

Questions raised by a read-through of the backend. These are points to confirm or investigate; they are not all confirmed defects.

1. **Should public endpoints require authentication or access controls?** The Flask app enables unrestricted CORS, and `/process-video`, `/videos`, and `/outputs/...` have no authentication checks. The backend launcher also opens a public ngrok tunnel. Is it intentional that anyone who obtains the tunnel URL can submit videos, enumerate processed runs, and fetch their outputs? (`backend/api_server.py:103-104, 118-151, 154-192`; launcher: `backend/api_server.py` near the `__main__` block.)

2. **What upload limits should be enforced?** The endpoint accepts any allowed extension and saves the complete upload before processing, with no visible size, duration, or decoded-resolution limit. Could very large or malformed videos consume disk, memory, or processing time? Should the endpoint verify the actual media type as well as its filename extension? (`backend/api_server.py:154-180`; `backend/pipeline_runner.py:183-200`.)

3. **Should unreadable input videos fail explicitly?** `run_video()` reads frame dimensions and FPS immediately after `cv2.VideoCapture(video_path)` without checking `cap.isOpened()` or validating dimensions. What response and cleanup are expected when OpenCV cannot open a file or reports invalid metadata? (`backend/pipeline_runner.py:183-200`.)

4. **How should processing errors clean up partial artifacts?** If inference raises, `/process-video` returns a 500, but the uploaded file and any partially written output directory are not cleaned up in this route. Should failed-run files be removed or retained with an explicit failed status, and should the manifest record failures? (`backend/api_server.py:176-192`.)

5. **Is the turn classification rule validated for the intended enforcement use?** Without a junction zone, orientation evidence alone classifies turns; with a zone, zone overlap alone classifies turns. Has this fallback behavior and the overlap threshold been checked against labeled clips? (`backend/pipeline_runner.py`; `config.py`.)

6. **How are blink predictions supposed to be interpreted over time?** Each prediction sets a slot's `is_signal` directly from the current predicted class, then bike-level signaling is `any_signal`; the final result accumulates whether signaling occurred while turning. Is a single positive frame sufficient, or should the classifier confidence and a time window / minimum number of blinking frames be considered? (`backend/pipeline_runner.py:443-455`; `backend/pipeline_runner.py:464-472`.)

7. **Should OCR outputs be validated before they become plate numbers?** GOT-OCR output is uppercased and stripped to an allowlist, but `ocr_conf_thresh` is explicitly unused and no plate-format or confidence validation is applied. Can plausible but incorrect OCR text become the persisted plate number, and what country/region-specific plate formats should be accepted? (`backend/plate_ocr.py:139-146, 172-207`.)

8. **Is the fixed OCR scratch path safe for the deployment model?** Every OCR call writes the crop to `/tmp/_plate_detector_got_ocr_crop.jpg` and then reads that same path. The in-process pipeline lock serializes requests in one app process, but would multiple worker processes or backend instances race on this shared filename? (`config.py` `GOT_OCR_TMP_PATH`; `backend/plate_ocr.py:162-176`; `backend/backend_support.py` `_pipeline_lock`.)

9. **Can the video loop safely recover from tracker update failures?** On a tracker exception the loop executes `continue`, skipping rendering, writer output, and `P.end_frame()` for that iteration. Could this leave profiler frame state inconsistent or produce a video with dropped output frames? Should frame accounting and output behavior be finalized in a `finally` path? (`backend/pipeline_runner.py:261-268`.)

10. **Do runtime dependencies and comments match the active OCR implementation?** `pipeline_runner.py` comments describe EasyOCR, while `plate_ocr.py` loads GOT-OCR 2.0 through Transformers. Are the required Transformers/GOT-OCR dependencies and model download expectations documented and pinned for the Colab setup? (`backend/pipeline_runner.py:163-174`; `backend/plate_ocr.py:91-124`; `requirements.txt`.)

11. **Should results and manifest entries have a retention policy?** Successful outputs, bike/plate images, uploads, and the manifest accumulate on disk, and the manifest is rewritten as a whole for every successful run. What storage limit, deletion policy, or archival behavior is expected for a long-running service? (`backend/api_server.py:176-180, 226-247`; `backend/backend_support.py:15-32`.)
