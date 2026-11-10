"""
Safe Ride Vision — inference backend (Colab + ngrok)
=====================================================

Matches the contract your frontend's api.js already expects:

    POST {ngrok_url}/process-video   (multipart/form-data)
      "video"           -> the video file           [always]
      "mode"            -> "direct" | "junction"    [always]
      "fps"             -> float                    [only if mode == "junction"]
      "junction_count"  -> int                       [only if mode == "junction"]
      "junctions"       -> JSON array (see below)     [only if mode == "junction"]

    Response JSON:
      {
        "output_video_url": "/outputs/<video_base>/annotatedvideo.mp4",
        "csv_download_url": "/outputs/<video_base>/<video_base>_bikes.csv",
        "logs": [ {bike_id, plate_number, plate_detected, has_both_mirrors,
                    has_both_indicators, took_turn,
                    indicator_on_while_turning}, ... ],   // raw CSV rows, unchanged, kept for backward compat
        "bikes": [ {..same fields as logs.., "violation": bool,
                     "bike_image_url": "/outputs/.../bikes/<id>/annotated_frame.jpg" | null,
                     "plate_image_url": "/outputs/.../bikes/<id>/plate_crop.jpg" | null}, ... ],
        "summary": {total_bikes, turned, signaled_correctly, violations,
                     plates_read, plates_missing, compliance_rate}
      }
    `bikes` is one "case file" per bike — this is the one the frontend
    should render (photo + plate crop + verdict); `logs` stays around
    unmodified for any older reader that already expects the flat CSV rows.

WHAT THIS FILE DOES
--------------------
1. Saves the uploaded video to disk.
2. Turns the frontend's `junctions` (rectangle/circle/polygon, in native
   frame pixels) into the TURN_ZONES format `deepsortVideo.py` already
   knows how to rasterize and check bikes against.
   - mode == "direct"            -> TURN_ZONES = []  (angle-detector only,
                                     exactly the "no polygon" pipeline path)
   - mode == "junction" + zones  -> TURN_ZONES = [...]  (zone-overlap ALSO
                                     counts as a turn, on top of the angle
                                     detector — same OR logic already in
                                     deepsortVideo.py, nothing to change there)
3. Runs `deepsortVideo.run_video()` on the saved file.
4. Moves the processed .mp4 into a public "output_public" folder and
   serves it back over a static route, so the frontend can just point a
   <video> tag at output_video_url.
5. Parses the pipeline's per-frame CSV log into JSON so it can ride along
   in the same response (api.js already reads `data.logs`).
6. Opens an ngrok tunnel and prints the URL to paste into the frontend.

SETUP (run once in a Colab cell before this file):
    !pip install flask flask-cors pyngrok -q

This file expects `deepsortVideo.py`, `track.py`, the `deep_sort/` package,
`blinker`, and all the model weight files it references to already be in
place exactly as `deepsortVideo.py` was written for — this backend just
calls into that pipeline, it doesn't change how detection/tracking works.
"""

import os
import json
import time
import shutil
import traceback
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import (
    UPLOAD_DIR, OUTPUT_PUBLIC_DIR, RUN_LOG_DIR, MANIFEST_PATH,
    ALLOWED_VIDEO_EXT, BACKEND_HOST, BACKEND_PORT,
)

# The DeepSORT/turn/indicator pipeline uploaded earlier. Keep this file
# next to it (or on PYTHONPATH). Importing it loads YOLO / OSNet / the
# blink-detector / RAFT once, at process startup — NOT per-request — so
# requests after the first one aren't stuck waiting on model loading.
import deepsortVideo as pipeline

# ----------------------------------------------------------------------
# Folders
# ----------------------------------------------------------------------
OUTPUT_DIR = OUTPUT_PUBLIC_DIR

for d in (UPLOAD_DIR, OUTPUT_DIR, RUN_LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Keep run artifacts and the pipeline's profiler logs under the configured run log folder.
pipeline.LOG_DIR = str(RUN_LOG_DIR)

app = Flask(__name__, static_folder=None)
CORS(app)  # the ngrok URL is cross-origin from the frontend, so allow it

# ----------------------------------------------------------------------
# Manifest — one entry per processed video, in the order it was RECEIVED
# (i.e. upload time), so /videos can hand the frontend everything sorted
# the same way regardless of how long each video took to process.
# ----------------------------------------------------------------------
_manifest_lock = threading.Lock()  # process_video() can run from multiple request threads


def _load_manifest():
    if not MANIFEST_PATH.exists():
        return []
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _append_manifest(entry):
    with _manifest_lock:
        entries = _load_manifest()
        entries.append(entry)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
    return entry


# ----------------------------------------------------------------------
# Junction (frontend) -> Turn zone (pipeline) conversion
# ----------------------------------------------------------------------
def _junction_to_zone(j):
    """One entry of the frontend's `junctions` array -> one dict in the
    shape deepsortVideo.build_zone_mask() / draw_zones() expect.

    Frontend shapes: "rectangle" | "circle" | "polygon", box is always the
    bounding box in native frame pixels; points is only present (and only
    used) for polygons.
    """
    shape = j.get("shape")
    box = j.get("box") or {}

    if shape == "circle":
        x = float(box.get("x", 0))
        y = float(box.get("y", 0))
        w = float(box.get("width", 0))
        h = float(box.get("height", 0))
        cx, cy = x + w / 2.0, y + h / 2.0
        radius = max(w, h) / 2.0
        if radius <= 0:
            return None
        return {"type": "circle", "center": (cx, cy), "radius": radius}

    if shape == "polygon":
        pts = j.get("points") or []
        points = [(float(p["x"]), float(p["y"])) for p in pts]
        if len(points) < 3:
            return None  # degenerate, not enough vertices to fill
        return {"type": "polygon", "points": points}

    # default: "rectangle"
    x = float(box.get("x", 0))
    y = float(box.get("y", 0))
    w = float(box.get("width", 0))
    h = float(box.get("height", 0))
    if w <= 0 or h <= 0:
        return None
    return {"type": "rectangle", "points": [(x, y), (x + w, y + h)]}


def _parse_turn_zones(form):
    """[] whenever mode != 'junction' or nothing usable was sent — that's
    the signal deepsortVideo.py's TURN_ZONES already treats as "run on
    angle-based turn detection alone" (build_zone_mask returns None,
    zone_overlap_ratio is always 0.0)."""
    if form.get("mode", "direct") != "junction":
        return []

    raw = form.get("junctions")
    if not raw:
        return []

    try:
        junctions = json.loads(raw)
    except (TypeError, ValueError):
        return []

    zones = []
    for j in junctions:
        try:
            zone = _junction_to_zone(j)
        except (TypeError, ValueError, KeyError):
            zone = None
        if zone is not None:
            zones.append(zone)
    return zones


def _parse_log_file(log_path):
    """The pipeline's per-bike summary CSV (one row per bike, written once
    by finalize_track() -> run_video()) -> list[dict], so it can travel in
    the JSON response the same way api.js already expects (`data.logs`).

    Columns, straight from deepsortVideo.finalize_track():
      bike_id, plate_number, plate_detected, has_both_mirrors,
      has_both_indicators, took_turn, indicator_on_while_turning
    All values arrive as strings (this is a raw CSV split, no type
    coercion) — see _to_bool() below for turning the boolean-looking
    columns back into real booleans before they're used for anything
    logical (filtering, violation flags, counts)."""
    if not os.path.exists(log_path):
        return []

    with open(log_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    if len(lines) < 2:
        return []

    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) != len(header):
            continue  # skip a malformed row rather than 500 the whole response
        rows.append(dict(zip(header, parts)))
    return rows


def _to_bool(value):
    """CSV columns come back as the strings 'True'/'False' (or empty) —
    this turns any of the shapes finalize_track() might have written into
    an actual bool, instead of every non-empty string (including the
    literal text 'False') being truthy."""
    return str(value).strip().lower() in ("true", "1", "yes")


# ----------------------------------------------------------------------
# Per-bike CSV row -> a full "case file" the frontend can render as a
# card: the two crops finalize_track()/_save_bike_snapshot() already put
# on disk at
#   <video_dir>/bikes/<track_id>/annotated_frame.jpg   (bike, boxed)
#   <video_dir>/bikes/<track_id>/plate_crop.jpg        (plate, if ever found)
# turned into public /outputs/... URLs (both live under OUTPUT_DIR, same
# as the video, so the existing /outputs/<path:filename> static route
# already serves them — nothing new to expose there), plus a single
# derived `violation` flag: this bike turned AND did NOT have its
# indicator on while doing it. Everything else about "why" (mirrors seen,
# plate read or not) rides along unchanged so the UI can explain the verdict.
# ----------------------------------------------------------------------
def _bike_case_files(rows, images_dir, output_dir):
    """`rows` are raw CSV dicts, keyed by the ACTUAL header run_video()
    writes to the summary CSV — see the csv_writer.writerow(header) call
    near the end of run_video():
        bike_id, plate_number, plate_detected, has_both_mirrors,
        has_both_indicators, took_turn, indicator_on_while_turning
    (note this is NOT the same naming as finalize_track()'s in-memory
    record dict — those get renamed on the way into the CSV). Mapping is
    done explicitly here, once, so everything downstream (and the
    frontend) can use one consistent set of names regardless of what the
    CSV itself calls them."""
    images_dir = Path(images_dir)
    cases = []
    for row in rows:
        track_id = row.get("bike_id", "")
        bike_folder = images_dir / str(track_id)
        ann_path = bike_folder / "annotated_frame.jpg"
        plate_path = bike_folder / "plate_crop.jpg"

        ever_turned = _to_bool(row.get("took_turn"))
        signaled = _to_bool(row.get("indicator_on_while_turning"))
        plate_number = (row.get("plate_number") or "").strip()

        cases.append({
            "track_id": track_id,
            "plate_number": plate_number,
            "plate_detected": _to_bool(row.get("plate_detected")),
            "mirror_seen_both": _to_bool(row.get("has_both_mirrors")),
            "indicator_seen_both": _to_bool(row.get("has_both_indicators")),
            "ever_turned": ever_turned,
            "ever_signaled_while_turning": signaled,
            # The one flag the UI needs to color a card red/green without
            # re-deriving the rule itself: turned without signaling.
            "violation": ever_turned and not signaled,
            "bike_image_url": (
                f"/outputs/{ann_path.relative_to(output_dir).as_posix()}"
                if ann_path.exists() else None
            ),
            "plate_image_url": (
                f"/outputs/{plate_path.relative_to(output_dir).as_posix()}"
                if plate_path.exists() else None
            ),
        })
    return cases


def _run_summary(cases):
    """Small aggregate block so the frontend can show headline stats
    (a stat strip) without recomputing them client-side from `bikes`."""
    total = len(cases)
    violations = sum(1 for c in cases if c["violation"])
    turned = sum(1 for c in cases if c["ever_turned"])
    signaled_correctly = sum(1 for c in cases if c["ever_turned"] and c["ever_signaled_while_turning"])
    plates_read = sum(1 for c in cases if c["plate_number"])
    return {
        "total_bikes": total,
        "turned": turned,
        "signaled_correctly": signaled_correctly,
        "violations": violations,
        "plates_read": plates_read,
        "plates_missing": total - plates_read,
        "compliance_rate": round((turned - violations) / turned, 3) if turned else None,
    }


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": time.time()})


@app.route("/outputs/<path:filename>", methods=["GET"])
def serve_output(filename):
    """Public static access to processed videos. api.js resolves
    output_video_url against the same ngrok base, so a relative path
    like '/outputs/xyz.mp4' just works."""
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/videos", methods=["GET"])
def list_videos():
    """All processed videos, sorted by the time they were RECEIVED
    (upload time) — not by when processing finished, so a slow big
    upload doesn't jump ahead of / behind a quick small one that came in
    after it.

    Query params:
      order   "asc" (default, oldest received first) | "desc" (newest first)
      limit   optional int — only return the last N entries after sorting
    """
    order = request.args.get("order", "asc").lower()
    entries = sorted(_load_manifest(), key=lambda e: e.get("received_at", 0))
    if order == "desc":
        entries = list(reversed(entries))

    limit = request.args.get("limit")
    if limit:
        try:
            entries = entries[: int(limit)]
        except ValueError:
            pass  # bad limit value — just ignore it rather than 400 the whole list

    return jsonify({"count": len(entries), "order": order, "videos": entries})


@app.route("/process-video", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return jsonify({"error": "No 'video' file in request."}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    ext = Path(video_file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        return jsonify({"error": f"Unsupported video type '{ext}'."}), 400

    mode = request.form.get("mode", "direct")

    # Captured the moment the upload lands — this, not "when processing
    # finished", is what defines "receiving order" for /videos below.
    received_at = time.time()

    # Timestamp-prefixed so concurrent uploads never collide, and so the
    # pipeline's own log filename (derived from the video's basename)
    # stays unique too.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_name = secure_filename(video_file.filename) or "video.mp4"
    in_name = f"{stamp}_{safe_name}"
    in_path = UPLOAD_DIR / in_name
    video_file.save(str(in_path))

    # This is the "if polygon is available or not" switch: junction
    # zones from the drawn shapes, or [] to fall back to angle-only turn
    # detection when the user ran in direct mode.
    turn_zones = _parse_turn_zones(request.form)
    pipeline.TURN_ZONES = turn_zones

    try:
        result = pipeline.run_video(str(in_path), str(OUTPUT_DIR))
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Pipeline failed: {exc}"}), 500

    # run_video() writes everything this run produces into its own
    # subfolder — OUTPUT_DIR/<video_base>/annotatedvideo.mp4,
    # OUTPUT_DIR/<video_base>/<video_base>_bikes.csv, etc. — and returns
    # the exact paths it used. Trust that return value instead of
    # re-deriving a flat OUTPUT_DIR/<in_name> path, which doesn't match
    # what the pipeline actually wrote and was causing every request to
    # 500 with "no output video was found" even on a successful run.
    produced_path = Path(result["output_video_path"])
    if not produced_path.exists():
        return jsonify({"error": "Pipeline finished but no output video was found."}), 500

    # The static /outputs/<path:filename> route serves anything under
    # OUTPUT_DIR, subfolders included, so hand back the path relative to
    # OUTPUT_DIR (e.g. "20260913-174110_041964/annotatedvideo.mp4").
    relative_video_path = produced_path.relative_to(OUTPUT_DIR).as_posix()

    # Real CSV path, straight from the pipeline — not a guessed
    # RUN_LOG_DIR/<name>.txt that was never where it got written.
    log_path = Path(result["summary_csv_path"])
    logs = _parse_log_file(log_path)

    # `bikes` = the same rows as `logs`, but each one turned into a full
    # case file (image URLs + a ready-made violation flag). `logs` is kept
    # exactly as before for anything already reading it — this is
    # additive, not a breaking change to the response shape.
    bikes = _bike_case_files(logs, result["video_images_dir"], OUTPUT_DIR)
    summary = _run_summary(bikes)

    # The CSV itself lives under OUTPUT_DIR too, so it's already servable
    # by the same static /outputs/<path:filename> route — just needs its
    # relative URL handed back so the frontend can offer a raw download.
    csv_download_url = f"/outputs/{log_path.relative_to(OUTPUT_DIR).as_posix()}" if log_path.exists() else None

    entry = {
        "id": in_name,                                    # stable key, doubles as the saved filename
        "original_filename": video_file.filename,
        "output_video_url": f"/outputs/{relative_video_path}",
        "csv_download_url": csv_download_url,
        "mode": mode,
        "junction_count": len(turn_zones),
        "received_at": received_at,                        # epoch seconds — sort key for /videos
        "received_at_iso": datetime.fromtimestamp(received_at, tz=timezone.utc).isoformat(),
        "log_row_count": len(logs),
        "summary": summary,                                 # so /videos can show headline stats too, not just processing time
        # Full per-bike case files (image URLs + violation flag included) —
        # stored here, not recomputed on demand, so a video reopened from
        # "View results" gets the exact same Analysis/Bikes deep-dive as the
        # one shown right after processing, with no extra request needed.
        "bikes": bikes,
    }
    _append_manifest(entry)

    return jsonify({
        "output_video_url": entry["output_video_url"],
        "csv_download_url": csv_download_url,
        "logs": logs,
        "bikes": bikes,
        "summary": summary,
        "mode": mode,
        "junction_count": len(turn_zones),
        "received_at": entry["received_at"],
    })


if __name__ == "__main__":
    from pyngrok import conf, ngrok

    # The Colab notebook passes this in the child process environment.
    # Setting the token only in the notebook kernel does not configure this
    # separate Python process.
    ngrok_token = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if not ngrok_token:
        raise RuntimeError(
            "A valid ngrok authtoken is required for a public Colab URL. "
            "Set NGROK_AUTHTOKEN in the notebook before starting the backend."
        )
    conf.get_default().auth_token = ngrok_token

    public_url = ngrok.connect(BACKEND_PORT, "http")
    print(f" * ngrok tunnel:  {public_url}")
    print(f" * Paste this into the frontend's backend-URL field: {public_url}")

    app.run(host=BACKEND_HOST, port=BACKEND_PORT)
