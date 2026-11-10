# vim: expandtab:ts=4:sw=4

from collections import defaultdict, deque
import numpy as np;
import traceback


def _log_track_error(track_id, context_msg):
    """Print full traceback + which track it happened on, flushed
    immediately, so a crash here shows up even if the process dies
    right after."""
    print(f"❌ EXCEPTION in track.py ({context_msg}) for track_id={track_id}", flush=True)
    traceback.print_exc()

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.is_turning =False 
        self.turn_angle = 0.0;
        self.box_color = (0, 255, 0);  # default green — straight, until first detect_turn()/update_box_color() call
        self.zone_overlap_ratio = 0.0  # fraction (0-1) of this bike's box inside any configured turn zone, this frame
        self.state = TrackState.Tentative
        self.trajectories =  deque(maxlen=100)  # expanded: turning bikes avg 64 frames, 59% need 51+
        self.features = []
        self.orientation = {"side":0 , "front-back":0}

        # ------------------------------------------------------------------
        # Mirrors — tracked with the SAME persistence trick as the indicator
        # slots below: two independent slots (left/right), each carrying its
        # last-known box forward by the bike's own motion whenever YOLO
        # misses it this frame, instead of losing/forgetting it. That keeps
        # "does this bike have two mirrors" a stable, noise-tolerant fact
        # rather than something that flickers on/off every time the mirror
        # detector has one bad frame (glare, motion blur, partial occlusion,
        # a car passing behind it, etc.). `hits` counts REAL detections only
        # (not carried-forward frames), so a single stray box can't flip the
        # "seen both mirrors" fact — see update_mirrors() / mirror_seen_both.
        # ------------------------------------------------------------------
        self.mirror_slots = [
            {"cord": None, "missed": 0, "hits": 0},
            {"cord": None, "missed": 0, "hits": 0},
        ]
        self.mirror_seen_both = False  # sticky: True forever once confidently proven

        self.trajectories =  deque(maxlen=100)  # expanded: turning bikes avg 64 frames, 59% need 51+
        self.features = []
        # Two independent indicator slots (e.g. left/right). Detections are
        # matched to whichever slot's last-known position is nearest, so a
        # box can't hop from one slot to the other and swap identities.
        # Each slot keeps its own image history (for blink detection) and
        # its own "missed" counter — when the indicator model doesn't fire
        # for a slot on a given frame, that slot's cord is carried forward
        # by the bike's own movement instead of going blank (see
        # update_indicators()).
        self.indicator_slots = [
            {"cord": None, "missed": 0, "history": deque(maxlen=30), "is_signal": False, "hits": 0},
            {"cord": None, "missed": 0, "history": deque(maxlen=30), "is_signal": False, "hits": 0},
        ]
        self.indicator_seen_both = False  # sticky: True forever once confidently proven

        self.flow_history = deque(maxlen=100)  # matches trajectories maxlen — same per-frame cadence
        self.is_signal = False;

        # ------------------------------------------------------------------
        # Largest-ever bike crop. Small/far/noisy detections make for
        # unreadable plate crops, so instead of running plate-OCR on every
        # frame we just keep whichever crop had the biggest bounding-box
        # area and OCR that ONE image, once, when the track ends. See
        # update_best_image().
        # ------------------------------------------------------------------
        self.best_image = None
        self.best_image_area = 0

        # ------------------------------------------------------------------
        # Full frame + this bike's box on that frame, captured at the same
        # moment as best_image (i.e. whenever best_image_area grows). Kept
        # separately from best_image (which stays a tight bike crop, used
        # for OCR) because the final saved-to-disk image should show the
        # WHOLE frame with the box drawn on it, not just the crop.
        # ------------------------------------------------------------------
        self.full_frame_image = None
        self.full_frame_box = None   # (x1, y1, x2, y2) on full_frame_image

        # Resolved as soon as OCR succeeds (no longer only at track end) —
        # see plate_detector.get_plate_from_precropped_bike().
        self.plate_text = None
        # Sticky: once a real (non-empty) plate reading is obtained, this
        # locks True and no further OCR attempts are made for this track —
        # every new "largest" crop before that point DOES get OCR'd though
        # (see update_best_image()'s return value / deepsortVideo.py).
        self.plate_locked = False
        # True the moment the plate DETECTOR (YOLO) finds a plate box on
        # this bike at least once, REGARDLESS of whether OCR ever manages
        # to read text off of it. So plate_detected=True, plate_text=""
        # is a valid/expected combo: YOLO saw a plate, OCR just couldn't
        # read it. Set in deepsortVideo.py's main loop, alongside the
        # plate DETECTOR + live-OCR block.
        self.plate_detected = False

        # Lifetime facts used for the final per-bike CSV summary row.
        # `ever_turned` / `ever_signaled_while_turning` are sticky (once
        # True, stay True) so a single frame is enough to record the fact,
        # matching how "did this bike turn" is really an event, not a
        # per-frame state.
        self.ever_turned = False
        self.ever_signaled_while_turning = False

        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        self.summary = "";

    def setTrajectories(self, bbox):
        """bbox = [x1, y1, x2, y2] — center point nikaalta hai aur store karta hai"""
        try:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            print("(CX, Cy): " ,cx , "," ,cy, flush=True );
            self.trajectories.append((cx, cy))
        except Exception:
            _log_track_error(self.track_id, "setTrajectories")
    
    def getAllTrajectoriesOfX(self):
        return list(self.trajectories)

    def setFlow(self, flow_stats):
        """flow_stats = (dx, dy, magnitude, angle_deg) — one RAFT reading for this frame"""
        self.flow_history.append(flow_stats)

    def getFlowHistory(self):
        return list(self.flow_history)
    
    def setOrientation(self , orien):
        self.orientation[orien] = self.orientation.get(orien, 0)+1;

    def update_mirrors(self, detections, movement_dx=0.0, movement_dy=0.0,
                        max_missed=15, min_hits_for_both=3):
        """Assign this frame's mirror detections to the two persistent
        slots, exactly like update_indicators() does for indicators — see
        that method's docstring for the full rationale. The short version:
        when YOLO fails to find a mirror this frame (bad angle, glare,
        motion blur, brief occlusion), we don't throw the last-known box
        away — we shift it forward by however much the bike itself moved,
        so the "guess" tracks the bike instead of freezing or vanishing.

        Parameters
        ----------
        detections : list
            Mirror boxes matched to this bike THIS frame, each
            `[conf, x1, y1, x2, y2]`. Usually 0, 1, or 2 items.
        movement_dx, movement_dy : float
            How much the bike itself moved since the last frame.
        max_missed : int
            After this many consecutive misses, a slot is cleared so a
            genuinely new mirror detection (e.g. after occlusion ends) can
            claim it fresh.
        min_hits_for_both : int
            A slot only counts toward "this bike has both mirrors" once it
            has been confirmed by this many REAL (non-carried-forward)
            detections — filters out a single noisy false-positive box
            flipping that fact on.
        """
        try:
            self._update_mirrors_impl(detections, movement_dx, movement_dy, max_missed, min_hits_for_both)
        except Exception:
            _log_track_error(self.track_id, "update_mirrors")

    def _update_mirrors_impl(self, detections, movement_dx, movement_dy, max_missed, min_hits_for_both):
        # 1. Carry every currently-known slot forward by the bike's motion.
        for slot in self.mirror_slots:
            if slot["cord"] is not None:
                c = slot["cord"]
                slot["cord"] = [c[0], c[1] + movement_dx, c[2] + movement_dy,
                                 c[3] + movement_dx, c[4] + movement_dy]

        def center(box):
            return ((box[1] + box[3]) / 2.0, (box[2] + box[4]) / 2.0)

        matched_slots = set()
        matched_dets = set()

        occupied = [i for i, s in enumerate(self.mirror_slots) if s["cord"] is not None]

        # 2. Match new detections to their nearest EXISTING (occupied) slot
        #    first, so a mirror can't hop slots and swap identity.
        if occupied and detections:
            pairs = []
            for si in occupied:
                scx, scy = center(self.mirror_slots[si]["cord"])
                for di, det in enumerate(detections):
                    dcx, dcy = center(det)
                    dist = ((scx - dcx) ** 2 + (scy - dcy) ** 2) ** 0.5
                    pairs.append((dist, si, di))
            pairs.sort(key=lambda p: p[0])
            for dist, si, di in pairs:
                if si in matched_slots or di in matched_dets:
                    continue
                matched_slots.add(si)
                matched_dets.add(di)
                slot = self.mirror_slots[si]
                slot["cord"] = detections[di]
                slot["missed"] = 0
                slot["hits"] += 1

        # 3. Leftover detections fill any empty slot first.
        empty = [i for i, s in enumerate(self.mirror_slots) if i not in matched_slots and s["cord"] is None]
        leftover_dets = [d for d in range(len(detections)) if d not in matched_dets]
        for di in leftover_dets:
            if not empty:
                break
            si = empty.pop(0)
            self.mirror_slots[si]["cord"] = detections[di]
            self.mirror_slots[si]["missed"] = 0
            self.mirror_slots[si]["hits"] += 1
            matched_dets.add(di)
            matched_slots.add(si)

        # 4. Any slot not matched this frame missed a beat — its cord was
        #    already carried forward by movement in step 1.
        for i, slot in enumerate(self.mirror_slots):
            if i in matched_slots:
                continue
            if slot["cord"] is not None:
                slot["missed"] += 1
                if slot["missed"] > max_missed:
                    slot["cord"] = None
                    slot["missed"] = 0
                    slot["hits"] = 0

        # 5. Sticky lifetime fact: has this bike ever confidently shown BOTH
        #    mirrors at once? Once True it stays True for the life of the
        #    track — a mirror going missing later (occlusion, angle) doesn't
        #    erase the fact that the bike has two.
        if not self.mirror_seen_both:
            h0 = self.mirror_slots[0]["hits"]
            h1 = self.mirror_slots[1]["hits"]
            if h0 >= min_hits_for_both and h1 >= min_hits_for_both:
                self.mirror_seen_both = True

    def update_indicators(self, detections, movement_dx=0.0, movement_dy=0.0,
                           max_missed=15, min_hits_for_both=3):
        """Assign this frame's indicator detections to the two persistent
        slots, keeping each slot's identity stable across frames.

        Parameters
        ----------
        detections : list
            Indicator boxes matched to this bike THIS frame, each
            `[conf, x1, y1, x2, y2]`. Usually 0, 1, or 2 items.
        movement_dx, movement_dy : float
            How much the bike itself moved since the last frame (e.g. the
            change in its trajectory center). Any slot that does NOT get a
            fresh detection this frame has its last-known cord shifted by
            this amount, so a briefly-missed indicator keeps tracking the
            bike instead of freezing in place or vanishing.
        max_missed : int
            After this many consecutive misses, a slot is cleared out so a
            genuinely new indicator (e.g. after occlusion ends) can claim it.
        min_hits_for_both : int
            A slot only counts toward "this bike has both indicators" once
            it has been confirmed by this many REAL detections — filters
            out a single noisy false-positive flipping that fact on.
        """
        try:
            self._update_indicators_impl(detections, movement_dx, movement_dy, max_missed, min_hits_for_both)
        except Exception:
            _log_track_error(self.track_id, "update_indicators")

    def _update_indicators_impl(self, detections, movement_dx, movement_dy, max_missed, min_hits_for_both):
        # 1. Carry every currently-known slot forward by the bike's motion.
        #    If a slot gets a real detection below, this gets overwritten;
        #    if not, this IS the slot's position for the frame.
        for slot in self.indicator_slots:
            if slot["cord"] is not None:
                c = slot["cord"]
                slot["cord"] = [c[0], c[1] + movement_dx, c[2] + movement_dy,
                                 c[3] + movement_dx, c[4] + movement_dy]

        def center(box):
            return ((box[1] + box[3]) / 2.0, (box[2] + box[4]) / 2.0)

        matched_slots = set()
        matched_dets = set()

        occupied = [i for i, s in enumerate(self.indicator_slots) if s["cord"] is not None]

        # 2. Match new detections to their nearest EXISTING (occupied) slot
        #    first — this is what keeps the two indicators from switching:
        #    a box near the last-known "slot 0" position stays slot 0 even
        #    if slot 1's detection happens to come first in the list.
        if occupied and detections:
            pairs = []
            for si in occupied:
                scx, scy = center(self.indicator_slots[si]["cord"])
                for di, det in enumerate(detections):
                    dcx, dcy = center(det)
                    dist = ((scx - dcx) ** 2 + (scy - dcy) ** 2) ** 0.5
                    pairs.append((dist, si, di))
            pairs.sort(key=lambda p: p[0])
            for dist, si, di in pairs:
                if si in matched_slots or di in matched_dets:
                    continue
                matched_slots.add(si)
                matched_dets.add(di)
                slot = self.indicator_slots[si]
                slot["cord"] = detections[di]
                slot["missed"] = 0
                slot["hits"] += 1

        # 3. Leftover detections (no occupied slot was close, or slot was
        #    empty to begin with) fill any empty slot first.
        empty = [i for i, s in enumerate(self.indicator_slots) if i not in matched_slots and s["cord"] is None]
        leftover_dets = [d for d in range(len(detections)) if d not in matched_dets]
        for di in leftover_dets:
            if not empty:
                break
            si = empty.pop(0)
            self.indicator_slots[si]["cord"] = detections[di]
            self.indicator_slots[si]["missed"] = 0
            self.indicator_slots[si]["hits"] += 1
            matched_dets.add(di)
            matched_slots.add(si)

        # 4. Any slot not matched this frame missed a beat — its cord was
        #    already carried forward by movement in step 1.
        for i, slot in enumerate(self.indicator_slots):
            if i in matched_slots:
                continue
            if slot["cord"] is not None:
                slot["missed"] += 1
                if slot["missed"] > max_missed:
                    slot["cord"] = None
                    slot["missed"] = 0
                    slot["is_signal"] = False
                    slot["hits"] = 0

        # 5. Sticky lifetime fact: has this bike ever confidently shown BOTH
        #    indicators at once? Mirrors mirror_seen_both's logic exactly.
        if not self.indicator_seen_both:
            h0 = self.indicator_slots[0]["hits"]
            h1 = self.indicator_slots[1]["hits"]
            if h0 >= min_hits_for_both and h1 >= min_hits_for_both:
                self.indicator_seen_both = True

    def setIndicatorImage(self, slot_idx, img_array):
        """Store the latest cropped indicator image for one slot (left/right
        kept separate). Only the last 30 crops per slot are kept."""
        if img_array is None:
            return
        self.indicator_slots[slot_idx]["history"].append(np.asarray(img_array))

    def getIndicatorImages(self, slot_idx):
        """Returns list of last 30 indicator images for the given slot."""
        return list(self.indicator_slots[slot_idx]["history"])

    def update_best_image(self, crop, min_area=6000, full_frame=None, box=None):
        """Keep the single largest bike crop seen so far for this track.

        Every frame the caller passes in this bike's current crop; if its
        area beats the previous best, it replaces it (old, smaller image is
        simply dropped). Crops below `min_area` are ignored outright — a
        bike that's too small/far/blurry to plausibly show a readable plate
        is noise, and letting it win "largest so far" just wastes the
        eventual OCR call and produces garbage text. Tune `min_area` to
        your footage (bigger for a plate to actually be legible in the
        crop); it's a raw pixel-area floor, not relative to the frame.

        Parameters
        ----------
        full_frame : ndarray, optional
            The complete video frame this crop was taken from. Stored
            alongside the crop (NOT re-cropped) so the final on-disk image
            can show the whole frame with the box drawn on it.
        box : tuple, optional
            (x1, y1, x2, y2) — this bike's box on `full_frame`.

        Returns
        -------
        bool
            True if `crop` just became the new best (largest-ever) image
            for this track. Callers use this to trigger plate OCR
            immediately on every new "largest" image instead of waiting
            until the track ends — see deepsortVideo.py.
        """
        try:
            return self._update_best_image_impl(crop, min_area, full_frame, box)
        except Exception:
            _log_track_error(self.track_id, "update_best_image")
            return False

    def _update_best_image_impl(self, crop, min_area, full_frame, box):
        if crop is None or crop.size == 0:
            return False

        h, w = crop.shape[:2]
        area = w * h
        if area < min_area:
            return False

        if area > self.best_image_area:
            self.best_image = crop.copy()
            self.best_image_area = area
            if full_frame is not None:
                self.full_frame_image = full_frame.copy()
            if box is not None:
                self.full_frame_box = tuple(map(int, box))
            return True

        return False

    def mark_turn_frame(self):
        """Call once per frame, after `is_turning` and `is_signal` are both
        finalized for the frame (i.e. after detect_turn()/the turn-zone
        check AND the indicator/blink check have both run). Rolls those
        per-frame flags up into the two sticky, track-lifetime facts used
        in the final CSV summary:

            ever_turned                  -> this bike turned at some point
            ever_signaled_while_turning  -> ...and used its indicator for
                                             at least one of those turns
        """
        if self.is_turning:
            self.ever_turned = True
            if self.is_signal:
                self.ever_signaled_while_turning = True

    def clearCords(self):
        """Deprecated no-op, kept only so any old caller that still invokes
        it doesn't crash. Mirrors and indicators are no longer wiped every
        frame — both now persist across missed frames on purpose (see
        update_mirrors() / update_indicators()); clearing them here would
        defeat that."""
        pass

    def GetSummary(self):
        turn_str = f" TURNING {self.turn_angle:.0f}°" if self.is_turning else ""
        turn_str += "|" + self.summary;
        return f"ID:{self.track_id}{turn_str}"
    
    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
    
    def detect_turn(self, angle_threshold=55, min_points=10, min_displacement=20):
        """Detect whether this track is making a turn.

        Classification is based solely on cumulative absolute angle change.
        A value above `angle_threshold` (default 30°, per report) strongly
        indicates a turn; 57% of turning bikes exceed this vs only 11% of
        straight-moving bikes.

        Parameters
        ----------
        angle_threshold : float
            Minimum cumulative absolute angle change (°) to flag a turn.
            Report recommends 30°.
        min_points : int
            Minimum trajectory points required before any decision is made.
        min_displacement : float
            Minimum total displacement (px) to rule out a stationary track.
        """
        try:
            return self._detect_turn_impl(angle_threshold, min_points, min_displacement)
        except Exception:
            _log_track_error(self.track_id, "detect_turn")
            self.is_turning = False
            self.turn_angle = 0.0
            return False, 0.0, ""

    def _detect_turn_impl(self, angle_threshold, min_points, min_displacement):
        pts = list(self.trajectories)
        seq_len = len(pts)

        # ── 1. Not enough history ────────────────────────────────────────────
        if seq_len < min_points:
            self.is_turning = False
            self.turn_angle = 0.0
            return False, 0.0, ""

        arr = np.array(pts, dtype=np.float32)

        # ── 2. Smoothing — 5-point moving average ────────────────────────────
        if len(arr) >= 5:
            kernel = np.ones(5, dtype=np.float32) / 5.0
            x_smooth = np.convolve(arr[:, 0], kernel, mode="same")
            y_smooth = np.convolve(arr[:, 1], kernel, mode="same")
            arr = np.stack([x_smooth, y_smooth], axis=1)

        # ── 3. Displacement guard ────────────────────────────────────────────
        total_disp = float(np.linalg.norm(arr[-1] - arr[0]))
        if total_disp < min_displacement:
            self.is_turning = False
            self.turn_angle = 0.0
            s = (f" - Track_id:{self.track_id} | CumAngle:0.00 | SeqLen:{seq_len}"
                 f" | TotalDisp:{total_disp:.1f} | Is Turning:False [low disp]")
            return False, 0.0, s

        # ── 4. Primary feature: cumulative absolute angle change ─────────────
        # Compute heading for each consecutive pair of trajectory points, then
        # sum the absolute frame-to-frame heading differences.  This directly
        # mirrors the "total absolute angle change" feature from the report
        # and is far more sensitive than the single 3-point angle used before.
        vectors = arr[1:] - arr[:-1]                          # (N-1, 2)
        norms   = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Filter out near-zero motion segments to avoid noisy headings
        valid_mask = (norms[:, 0] > 1e-2)
        if valid_mask.sum() < 2:
            self.is_turning = False
            self.turn_angle = 0.0
            s = (f" - Track_id:{self.track_id} | CumAngle:0.00 | SeqLen:{seq_len}"
                 f" | TotalDisp:{total_disp:.1f} | Is Turning:False [no valid motion]")
            return False, 0.0, s

        valid_vecs = vectors[valid_mask]
        # Heading angles in degrees
        headings = np.degrees(np.arctan2(valid_vecs[:, 1], valid_vecs[:, 0]))
        # Absolute frame-to-frame heading differences (wrapped to [-180, 180])
        delta = np.diff(headings)
        delta = (delta + 180) % 360 - 180                     # wrap
        cumulative_angle_change = float(np.sum(np.abs(delta)))

        # ── 5. Classification: cumulative angle threshold only ───────────────
        self.turn_angle = cumulative_angle_change
        if cumulative_angle_change <= 10:
            self.summary = "A<10"
            self.is_turning = False;
        elif self.orientation["side"] >=3 and self.orientation["front-back"] >=3   and cumulative_angle_change > 15: 
            self.summary = "A>10+side"
            self.is_turning = True;

        # NOTE: box_color is no longer decided here. A turn-zone check (see
        # deepsortVideo.py) can also flag is_turning AFTER this method
        # returns, so setting color here would get overwritten/stale. Color
        # is now decided once per frame, after every turn signal (angle-based
        # AND zone-based) is finalized, via update_box_color() below.

        s = (f" - Track_id:{self.track_id}"
             f" | CumAngle:{cumulative_angle_change:.2f}"
             f" | Threshold:{angle_threshold}"
             f" | SeqLen:{seq_len}"
             f" | TotalDisp:{total_disp:.1f}"
             f" | Is Turning:{self.is_turning}")

        print(
            "Track_id:", self.track_id,
            "| CumAngle:", round(cumulative_angle_change, 2),
            "| Threshold:", angle_threshold,
            "| Is Turning:", self.is_turning
        )
        return True, cumulative_angle_change, s

    def update_box_color(self):
        """Single source of truth for this bike's box color, called once per
        frame after `is_turning` (from detect_turn AND/OR a turn-zone
        overlap check) and `is_signal` are both finalized. Same rule applies
        whether turning was flagged by the angle-based detector alone, by a
        turn-zone overlap alone, or by both — the two paths just feed the
        same `is_turning` flag before this runs.

            not turning                    -> green  (straight)
            turning, indicator NOT active   -> red    (turning, no signal)
            turning, indicator active       -> black  (turning, signaled)
        """
        if not self.is_turning:
            self.box_color = (0, 255, 0)   # green (BGR) — straight
        elif self.is_signal:
            self.box_color = (0, 0, 0)     # black — turning, indicator ON
        else:
            self.box_color = (0, 0, 255)   # red — turning, indicator OFF/not detected
        return self.box_color
