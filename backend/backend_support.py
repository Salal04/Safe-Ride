"""Pure helpers and manifest persistence for the Safe Ride API."""
import csv
import json
import os
import threading
from pathlib import Path

from config import MANIFEST_PATH

_manifest_lock = threading.Lock()  # process_video() can run from multiple request threads
# The inference module keeps TURN_ZONES and profiler/model state at module
# scope, so two requests must not enter run_video() at the same time.
_pipeline_lock = threading.Lock()


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

    rows = []
    with open(log_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Ignore malformed rows rather than failing the whole response.
            if row and None not in row:
                rows.append(row)
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


