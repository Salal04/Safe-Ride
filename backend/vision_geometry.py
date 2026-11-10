"""Frame geometry, optical-flow statistics, and junction-zone rasterization."""
import cv2
import numpy as np

from config import TURN_ZONE_DRAW_COLOR, TURN_ZONE_DRAW_THICKNESS

def bike_flow_stats(flow, bbox):
    """
    Average optical-flow vector inside a single bike's bbox for this frame.
    Returns (dx, dy, magnitude, angle_deg) — all zero if flow is unavailable
    (first frame) or the box is degenerate.
    """
    if flow is None:
        return 0.0, 0.0, 0.0, 0.0

    h, w = flow.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0, 0.0

    patch = flow[y1:y2, x1:x2]
    dx = float(np.mean(patch[..., 0]))
    dy = float(np.mean(patch[..., 1]))
    magnitude = float(np.hypot(dx, dy))
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return dx, dy, magnitude, angle


def xyxy_to_tlwh(box):
    # box = [x1, y1, x2, y2]
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return [int(x1), int(y1), int(w), int(h)]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def is_box_inside(box1, box2, method="center"):
    """
    Check if box2 is inside box1.

    Parameters:
        box1: list or tuple [x1, y1, x2, y2]  # parent box
        box2: list or tuple [x1, y1, x2, y2]  # child box
        method: "center" (default), "full", or "iou"
    
    Returns:
        True if box2 is considered inside box1, else False
    """

    x1, y1, x2, y2 = box1
    bx1, by1, bx2, by2 = box2

    if method == "center":
        # Check center point of box2
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        return x1 <= cx <= x2 and y1 <= cy <= y2

    elif method == "full":
        # Check full containment
        return x1 <= bx1 and y1 <= by1 and x2 >= bx2 and y2 >= by2

    elif method == "iou":
        # Check IoU threshold (partial overlap)
        inter_x1 = max(x1, bx1)
        inter_y1 = max(y1, by1)
        inter_x2 = min(x2, bx2)
        inter_y2 = min(y2, by2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (bx2 - bx1) * (by2 - by1)
        iou = inter_area / (area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0

        return iou > 0.1  # threshold adjustable

    else:
        raise ValueError("Method must be 'center', 'full', or 'iou'")

def _zone_polygon_points(zone):
    """Return an (N,2) int32 array of vertices for a polygon/rectangle zone.
    Rectangles given as 2 corner points get expanded to 4; anything else
    (>=3 points, arbitrary polygon) passes through as-is. Not used for
    circles, which are rasterized directly.
    """
    pts = zone["points"]
    if len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        x1, x2 = sorted((float(x1), float(x2)))
        y1, y2 = sorted((float(y1), float(y2)))
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return np.array(pts, dtype=np.int32)


def build_zone_mask(zones, width, height):
    """Rasterize every configured turn zone — any mix of polygon, rectangle,
    or circle, any count — into a single binary (H, W) mask where a pixel is
    1 if it falls inside ANY zone. Built once per video (zones are treated
    as static for the whole clip), so per-frame overlap checks below are
    just a cheap crop + count against this mask.

    Returns None if `zones` is empty, so callers can skip zone-based turn
    detection entirely when no zones are configured (the "no polygon" case).
    """
    if not zones:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    for zone in zones:
        ztype = zone.get("type", "polygon").lower()
        if ztype == "circle":
            cx, cy = map(int, zone["center"])
            r = int(zone["radius"])
            cv2.circle(mask, (cx, cy), r, 1, thickness=-1)
        elif ztype in ("polygon", "rectangle"):
            pts_arr = _zone_polygon_points(zone)
            cv2.fillPoly(mask, [pts_arr], 1)
        else:
            raise ValueError(f"Unknown turn zone type: {ztype!r} (expected 'polygon', 'rectangle', or 'circle')")
    return mask


def zone_overlap_ratio(mask, bbox):
    """Fraction (0.0-1.0) of `bbox` [x1,y1,x2,y2] that falls inside ANY
    configured turn zone, per the mask from build_zone_mask(). Returns 0.0
    if there's no mask (no zones configured) or the box is degenerate/fully
    off-frame.
    """
    if mask is None:
        return 0.0

    h, w = mask.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    box_area = (x2 - x1) * (y2 - y1)
    inside = int(np.count_nonzero(mask[y1:y2, x1:x2]))
    return inside / box_area if box_area > 0 else 0.0


def draw_zones(frame, zones, color=TURN_ZONE_DRAW_COLOR, thickness=TURN_ZONE_DRAW_THICKNESS):
    """Draw every configured turn zone onto `frame` — any mix of polygon,
    rectangle, or circle, any count — so the output video shows exactly what
    bikes are being checked against. No-op if `zones` is empty.
    """
    for zone in zones:
        ztype = zone.get("type", "polygon").lower()
        if ztype == "circle":
            cx, cy = map(int, zone["center"])
            r = int(zone["radius"])
            cv2.circle(frame, (cx, cy), r, color, thickness)
        elif ztype in ("polygon", "rectangle"):
            pts_arr = _zone_polygon_points(zone).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_arr], isClosed=True, color=color, thickness=thickness)
        else:
            raise ValueError(f"Unknown turn zone type: {ztype!r} (expected 'polygon', 'rectangle', or 'circle')")


