"""Detection association, bike track finalization, and public snapshots."""
import logging
import os
import traceback

import cv2

from config import *
from .pipeline_profiler import P
from .vision_geometry import compute_iou, is_box_inside

logger = logging.getLogger("deepsortVideo")


def log_exception(context_msg):
    logger.error("❌ EXCEPTION in %s", context_msg)
    logger.error(traceback.format_exc())
    for handler in logger.handlers:
        handler.flush()

def getDetections(model, frame , detect , confidence = MIN_CONFIDENCE, tag = "det.unknown"):
        try:
            with P.stage(f"{tag}.model_call"):        # preprocess + GPU inference + NMS, CUDA-synced
                results = model(frame, imgsz=INFERENCE_IMAGE_SIZE)
            try:
                P.note_ult(tag, results[0].speed)     # Ultralytics' own pre/infer/post ms
            except Exception:
                pass
        except Exception:
            log_exception(f"YOLO inference failed for detect={detect}")
            return [], []

        detections = []
        class_name = []
        with P.stage(f"{tag}.parse"):                 # python box loop + .cpu().numpy() + print()
            try:
                res = results[0]
                if hasattr(res, "boxes"):
                    for box in res.boxes:
                        try:
                            cls_id= int(box.cls[0]) if hasattr(box.cls, "__getitem__") else int(box.cls)
                            name = res.names[cls_id]
                            print(f'ClassName : give: {detect} -> found {name}')
                            mx1, my1, mx2, my2 = box.xyxy[0].cpu().numpy()
                            mx1, my1, mx2, my2 = int(mx1), int(my1), int(mx2), int(my2)
                            conf = float(box.conf[0].cpu().numpy()) if hasattr(box.conf, "__getitem__") else float(box.conf)
                            if conf > confidence and name in detect:
                                detections.append([conf, mx1, my1, mx2, my2])
                                class_name.append(name)
                        except Exception:
                            log_exception(f"parsing one YOLO box for detect={detect}")
                            continue
            except Exception:
                log_exception(f"parsing YOLO results for detect={detect}")

        return detections , class_name


def active_tracks(tracker):
    """
    Tracks that are confirmed (passed n_init) AND were actually matched to a
    detection this frame (time_since_update == 0).

    Iterating tracker.tracks directly (as the original code did) also pulls
    in tentative tracks (unconfirmed, can vanish next frame) and stale/ghost
    tracks that Deep SORT is still coasting on Kalman prediction alone for up
    to max_age frames after last seeing them. Drawing/associating against
    those is what produces "duplicate" boxes on a single physical object
    when a fresh track spawns for it while the old ghost track is still
    being rendered.
    """
    return [t for t in tracker.tracks if t.is_confirmed() and t.time_since_update == 0]


def best_match_track(tracks, box, method="center"):
    """
    Return the single track whose box has the highest IoU with `box` among
    those where is_box_inside(...) also holds, or None.

    The original code assigned a mirror/indicator/orientation detection to
    EVERY track that satisfied is_box_inside(). When two tracked bikes
    overlap (common in traffic), that lets one detection get attached to
    more than one track -> duplicate annotations that look like ID mixups.
    Picking a single best match fixes that.
    """
    best_track, best_iou = None, 0.0
    for t in tracks:
        t_box = t.to_tlbr()
        if not is_box_inside(t_box, box, method=method):
            continue
        iou = compute_iou(t_box, box)
        if iou >= best_iou:
            best_iou = iou
            best_track = t
    return best_track


def _save_bike_snapshot(track, images_dir):
    """Write (or overwrite) this bike's PUBLIC annotated frame to disk
    RIGHT NOW, using whatever track.full_frame_image / track.full_frame_box
    currently hold:
        images_dir/<track_id>/annotated_frame.jpg   (full frame + box —
            this is the final artifact a human/frontend sees)

    Called every time update_best_image() reports a NEW best (bigger) crop
    — i.e. continuously, throughout video processing, not just once when
    the track ends — so the file keeps getting overwritten by whichever
    frame gave the biggest view of this bike so far. Also called once more
    from finalize_track() as a safety net.

    The bike's plate_crop.jpg is a SEPARATE, independent file — see
    _save_plate_snapshot() below, driven by its own "biggest plate box
    seen so far" tracking (best_plate_area_by_track in run_video()), not
    tied to whichever frame happened to have the biggest overall bike crop.

    Returns the annotated_frame.jpg path, or None if there was nothing to
    save.
    """
    if images_dir is None or track.best_image is None:
        return None
    try:
        bike_dir = os.path.join(images_dir, str(track.track_id))
        os.makedirs(bike_dir, exist_ok=True)

        if track.full_frame_image is not None:
            annotated = track.full_frame_image.copy()
            if track.full_frame_box is not None:
                x1, y1, x2, y2 = track.full_frame_box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"ID:{track.track_id}", (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            annotated_frame_path = os.path.join(bike_dir, "annotated_frame.jpg")
            cv2.imwrite(annotated_frame_path, annotated)
            return annotated_frame_path

        return None
    except Exception as e:
        print(f"Track {track.track_id}: failed to save snapshot image ({e}).")
        return None


def _save_plate_snapshot(track_id, plate_crop_rgb, images_dir):
    """Write (or overwrite) this bike's PUBLIC plate crop to disk RIGHT NOW:
        images_dir/<track_id>/plate_crop.jpg

    Called every frame the plate DETECTOR (running live, inside the main
    loop — see run_video()) finds a bigger/clearer plate box for this bike
    than any seen before that video. NO OCR happens here or anywhere in the
    frame loop — this only ever saves the located CROP. Reading the actual
    numbers off of it is the post-loop pass's job, at the very end of
    run_video(), once every frame has been processed — see
    read_plate_text_from_image() in plate_detector.py.

    Parameters
    ----------
    plate_crop_rgb : np.ndarray
        RGB image. detect_plates_in_bike_crop() converts its input to RGB
        internally before slicing the plate out of it, so what it returns
        is RGB, not the BGR order cv2 normally uses — flip it back here
        before cv2.imwrite, or the saved jpg's colors come out swapped.

    Returns the plate_crop.jpg path, or None if there was nothing to save.
    """
    if images_dir is None or plate_crop_rgb is None or plate_crop_rgb.size == 0:
        return None
    try:
        bike_dir = os.path.join(images_dir, str(track_id))
        os.makedirs(bike_dir, exist_ok=True)
        plate_crop_bgr = cv2.cvtColor(plate_crop_rgb, cv2.COLOR_RGB2BGR)
        plate_crop_path = os.path.join(bike_dir, "plate_crop.jpg")
        cv2.imwrite(plate_crop_path, plate_crop_bgr)
        return plate_crop_path
    except Exception as e:
        print(f"Track {track_id}: failed to save plate crop ({e}).")
        return None


def finalize_track(track, records, images_dir=None):
    """Called exactly once per bike, the moment its track is gone for good
    — either DeepSORT actually deleted it (lost for too many frames), or
    the video ended while it was still alive.

    Both plate DETECTION and OCR already happened live, every frame this
    bike was on screen, in the main loop (see run_video()'s plate
    DETECTOR + live-OCR block) — so by the time this runs,
    images_dir/<track_id>/plate_crop.jpg already holds the biggest/
    clearest plate crop ever found for this bike, and track.plate_text
    already holds its OCR reading, if one was ever successfully obtained
    (track.plate_locked True). Nothing else needs to run here to resolve
    the plate — this function just carries that result into `records`.

    So this function:
      1. Re-saves this bike's PUBLIC annotated frame via
         _save_bike_snapshot() — images_dir/<track_id>/annotated_frame.jpg
         — as a safety net, in case the frame loop's own continuous saving
         never fired for this track.
      2. Appends one record — everything the final CSV row needs, including
         plate_number straight from track.plate_text (empty string if this
         bike's plate was never successfully read) — to `records`, a list
         shared across the whole run_video() call. Any track whose plate
         is still unresolved here gets one more OCR attempt in the
         post-loop fallback pass near the end of run_video(), reading
         straight off its saved plate_crop.jpg.

    Tracks that never produced a large-enough crop (see
    MIN_BIKE_AREA_FOR_BEST_IMAGE / Track.update_best_image) are skipped
    entirely and never get a record — a track that was always tiny was
    most likely a noisy/false detection, never a bike close enough to
    read a plate from, so it isn't worth a row.
    """
    if track.best_image is None:
        print(f"Track {track.track_id}: no large-enough crop ever captured — treating as noise, skipping.")
        return

    annotated_frame_path = _save_bike_snapshot(track, images_dir)
    if annotated_frame_path:
        print(f"Track {track.track_id}: snapshot image -> {annotated_frame_path}")

    records.append({
        "track_id": track.track_id,
        "plate_number": track.plate_text or "",
        # True if the plate DETECTOR (YOLO) ever found a plate box on this
        # bike at least once, regardless of whether OCR could read it —
        # see Track.plate_detected in track.py.
        "plate_detected": bool(getattr(track, "plate_detected", False)),
        "mirror_seen_both": track.mirror_seen_both,
        "indicator_seen_both": track.indicator_seen_both,
        "ever_turned": track.ever_turned,
        "ever_signaled_while_turning": track.ever_signaled_while_turning,
    })

    print(f"Track {track.track_id} finalized — plate={track.plate_text!r} "
          f"(locked={track.plate_locked}, detected={getattr(track, 'plate_detected', False)}), "
          f"both_mirrors={track.mirror_seen_both}, "
          f"both_indicators={track.indicator_seen_both}, turned={track.ever_turned}, "
          f"signaled_while_turning={track.ever_signaled_while_turning}")


def draw_box(image, coords, label="Box", color=(0,255,0)):
    x1, y1, x2, y2 = map(int, coords)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

