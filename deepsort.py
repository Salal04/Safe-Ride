"""
YOLOv8 (person-only) + original nwojke/deep_sort integration.

Usage:
    python yolo_deepsort.py --source 0            # webcam
    python yolo_deepsort.py --source input.mp4    # video file
"""

import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# --- Deep SORT (nwojke/deep_sort) imports ---
# Make sure `deep_sort` repo folder is in PYTHONPATH or installed as package
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet  # contains create_box_encoder

# ----------------- User params -----------------
YOLO_MODEL = "yolov10m.pt"                 # change if you want bigger model
REID_WEIGHTS = "deep_sort/networks/mars-small128.pb"  # put your reid model here
MAX_COSINE_DISTANCE = 0.2
NN_BUDGET = 100
MIN_CONFIDENCE = 0.3        # ignore detections below this
N_INIT = 3                  # confirm track after N frames
MAX_AGE = 30                # drop track after this many frames missing
DRAW_TRAILS = True
TRAIL_LEN = 20
Mirror_indicater_DETECTOR ="mirror-indicator-yolov10m\\weights\\bestM.pt"

# ------------------------------------------------

def xyxy_to_tlwh(box):
    # box = [x1, y1, x2, y2]
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return [int(x1), int(y1), int(w), int(h)]

def main(source=0, output=None, show=True):
    # --- Load YOLOv10 ---
    yolo = YOLO(YOLO_MODEL)

    # --- Init DeepSORT metric & tracker ---
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
    tracker = Tracker(metric, max_iou_distance=0.7, max_age=MAX_AGE, n_init=N_INIT)

    # --- Create feature encoder (uses TF frozen graph) ---
    if not Path(REID_WEIGHTS).exists():
        raise FileNotFoundError(f"ReID model not found at {REID_WEIGHTS}. Download 'mars-small128.pb' or equivalent and place it there.")
    encoder = gdet.create_box_encoder(REID_WEIGHTS, batch_size=1)

    # --- Video source ---
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mirror_indicator_detector = YOLO(Mirror_indicater_DETECTOR);

    writer = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    # For drawing trails
    trails = {}  # track_id -> deque of centroids

    frame_idx = 0
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # YOLO inference (single frame). results[0] is the Results object
            results = yolo(frame, imgsz=640, device=None)  # device None -> use default (GPU if available)
            res = results[0]

            # Collect motorcycle detections: [x1,y1,x2,y2,conf]
            motorcycle_boxes = []
            if hasattr(res, "boxes"):
                for box in res.boxes:
                    cls = int(box.cls.cpu().numpy()[0]) if hasattr(box, "cls") else int(box.cls)
                    MOTORCYCLE_CLASS_ID = 3  # apne data.yaml se confirm karo

                   

                    if cls != MOTORCYCLE_CLASS_ID:
                      continue


                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy()) if hasattr(box, "conf") else float(box.conf)
                    if conf < MIN_CONFIDENCE:
                        continue
                    motorcycle_boxes.append([float(x1), float(y1), float(x2), float(y2), float(conf)])

            # If no detections, still call predict/update with empty list
            detections = []
            if len(motorcycle_boxes) > 0:
                # Convert to tlwh required by encoder
                tlwhs = [xyxy_to_tlwh(p[:4]) for p in motorcycle_boxes]
                # Extract features for each bbox (encoder expects image in BGR and tlwh list)
                features = encoder(frame, tlwhs)  # list of feature vectors

                # Build Detection objects for Deep SORT
                for i, det in enumerate(motorcycle_boxes):
                    tlwh = tlwhs[i]
                    conf = det[4]
                    feature = features[i]
                    detections.append(Detection(tlwh, conf, feature))

            # --- Run tracker ---
            tracker.predict()
            tracker.update(detections)

            # --- Draw tracks ---
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue
                
                track_id = track.track_id
                bbox = track.to_tlbr()  # tlbr = [min x,min y,max x,max y]
                x1, y1, x2, y2 = map(int, bbox)

                motorcycle_crop = frame[y1:y2, x1:x2];

                mask_results = mirror_indicator_detector(motorcycle_crop); # YOLOv8 detect
                res = mask_results[0] 

                for box in res.boxes:
                    cls_id = int(box.cls[0])        
                    cls_name = res.names[cls_id]
                    mx1, my1, mx2, my2 = box.xyxy[0].cpu().numpy()
                    mx1, my1, mx2, my2 = int(mx1), int(my1), int(mx2), int(my2)
                    global_x1 = x1 + mx1
                    global_y1 = y1 + my1
                    global_x2 = x1 + mx2
                    global_y2 = y1 + my2

                    if "mirror" in cls_name or "no_mirror" in cls_name:
                        track.setMirror(cls_name, [ global_x1 ,global_y1 ,global_x2,global_y2 ])
                    #if "glass" in cls_name or "no_glass" in cls_name:
                        #track.setGlasses(cls_name, [ global_x1 ,global_y1 ,global_x2,global_y2 ]) 

                #draw mask
                if track.mirror_Cord is not None and len(track.mirror_Cord) > 0:
                    mx1, my1, mx2, my2 = map(int, track.mirror_Cord)
                    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0,255,0), 2)
                    cv2.putText(frame, track.mirror_status, (mx1, my1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                #draw Glasses    
                #if track.Glasses_Cord is not None and len(track.Glasses_Cord) > 0:
                    #gx1, gy1, gx2, gy2 = map(int, track.Glasses_Cord)
                    #cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255,0,0), 2)
                    #cv2.putText(frame, track.Glasses_status, (gx1, gy1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)   
                
                # color by track id
                color = ((track_id * 37) % 255, (track_id * 17) % 255, (track_id * 29) % 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, track.GetSummary(), (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # draw center point & trail
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if DRAW_TRAILS:
                    if track_id not in trails:
                        trails[track_id] = deque(maxlen=TRAIL_LEN)
                    trails[track_id].appendleft((cx, cy))

                    # draw trail
                    pts = list(trails[track_id])
                    for j in range(1, len(pts)):
                        cv2.line(frame, pts[j - 1], pts[j], color, 2)

            # show fps
            if frame_idx % 10 == 0:
                elapsed = time.time() - t0
                fps_est = frame_idx / elapsed if elapsed > 0 else 0.0
                cv2.putText(frame, f"FPS: {fps_est:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            if show:
                cv2.imshow("YOLOv10m+ DeepSORT", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if writer:
                writer.write(frame)

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="E:\\dataset\\projectCode\\deepSort\\vid1.mp4", help="0 for webcam or path to video")
    p.add_argument("--output", type=str, default=None, help="optional output video path")
    args = p.parse_args()

    src = int(args.source) if args.source.isnumeric() else args.source
    main(source=src, output=args.output, show=True)
