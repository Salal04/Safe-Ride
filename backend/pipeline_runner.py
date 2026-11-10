"""Video inference orchestration and frame processing."""
import csv
import gc
import logging
import os
import shutil
import subprocess
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import *
from deep_sort.deep_sort import nn_matching, preprocessing
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from .pipeline_profiler import P, _safe_slug
from .pipeline_models import blink_detector, build_osnet_encoder, device, extract_indicator_crop, predict_from_track
from .pipeline_tracking import (
    _save_bike_snapshot, _save_plate_snapshot, active_tracks, best_match_track,
    draw_box, finalize_track, getDetections,
)
from .plate_detector import (
    detect_plates_in_bike_crop, read_plate_text_from_image,
    warm_up_ocr, warm_up_plate_detector,
)
from .vision_geometry import bike_flow_stats, build_zone_mask, draw_zones, xyxy_to_tlwh, zone_overlap_ratio

logger = logging.getLogger("deepsortVideo")


def log_exception(context_msg):
    logger.error("❌ EXCEPTION in %s", context_msg)
    import traceback
    logger.error(traceback.format_exc())
    for handler in logger.handlers:
        handler.flush()


def run_video(video_path, output_path=None, display_name=None, turn_zones=None):
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
    turn_zones = TURN_ZONES if turn_zones is None else turn_zones

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

    # Built once — zones are static for the whole clip. None if turn_zones
    # is empty, in which case zone_overlap_ratio() always returns 0.0 and
    # the pipeline runs on angle-based turn detection alone (unchanged
    # behavior for videos with no zones configured).
    zone_mask = build_zone_mask(turn_zones, width, height)

    frame_idx = 0

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
                # enough. With turn_zones empty, zone_mask is None and this is a
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
        # No-op if turn_zones is empty — safe to always call.
        with P.stage("draw.zones"):
            draw_zones(frame, turn_zones)

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

