# vim: expandtab:ts=4:sw=4

from collections import deque
import numpy as np;
import traceback


def _log_track_error(track_id, context_msg):
    """Print full traceback + which track it happened on, flushed
    immediately, so a crash here shows up even if the process dies
    right after."""
    print(f"❌ EXCEPTION in track.py ({context_msg}) for track_id={track_id}", flush=True)
    traceback.print_exc()


class ComponentKalman:
    """Constant-velocity Kalman filter for one mirror/indicator center."""

    def __init__(self, center):
        self.state = np.array([center[0], center[1], 0.0, 0.0], dtype=np.float32)
        self.covariance = np.diag([25.0, 25.0, 100.0, 100.0]).astype(np.float32)
        self.transition = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.process_noise = np.diag([1.0, 1.0, 4.0, 4.0]).astype(np.float32)
        self.measurement_noise = np.diag([9.0, 9.0]).astype(np.float32)

    def predict(self):
        self.state = self.transition @ self.state
        self.covariance = (
            self.transition @ self.covariance @ self.transition.T + self.process_noise
        )
        return self.state[:2].copy()

    def measurement_distance(self, center):
        measurement = np.asarray(center, dtype=np.float32)
        residual = measurement - self.observation @ self.state
        innovation_covariance = (
            self.observation @ self.covariance @ self.observation.T
            + self.measurement_noise
        )
        return float(residual.T @ np.linalg.solve(innovation_covariance, residual))

    def update(self, center):
        measurement = np.asarray(center, dtype=np.float32)
        residual = measurement - self.observation @ self.state
        innovation_covariance = (
            self.observation @ self.covariance @ self.observation.T
            + self.measurement_noise
        )
        gain = self.covariance @ self.observation.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + gain @ residual
        identity = np.eye(4, dtype=np.float32)
        correction = identity - gain @ self.observation
        self.covariance = (
            correction @ self.covariance @ correction.T
            + gain @ self.measurement_noise @ gain.T
        )
        return self.state[:2].copy()

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
        self.box_color = (0, 255, 0);  # default green — straight, until a turn-zone/orientation result is available
        self.zone_overlap_ratio = 0.0  # fraction (0-1) of this bike's box inside any configured turn zone, this frame
        self.state = TrackState.Tentative
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
            {"cord": None, "missed": 0, "hits": 0, "kalman": None},
            {"cord": None, "missed": 0, "hits": 0, "kalman": None},
        ]
        self.mirror_seen_both = False  # sticky: True forever once confidently proven

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
            {"cord": None, "missed": 0, "history": deque(maxlen=30), "is_signal": False, "hits": 0, "kalman": None},
            {"cord": None, "missed": 0, "history": deque(maxlen=30), "is_signal": False, "hits": 0, "kalman": None},
        ]
        self.indicator_seen_both = False  # sticky: True forever once confidently proven

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
    def setOrientation(self , orien):
        self.orientation[orien] = self.orientation.get(orien, 0)+1;

    def update_mirrors(self, detections, movement_dx=0.0, movement_dy=0.0,
                        max_missed=15, min_hits_for_both=3):
        """Update mirror slots with gated Kalman prediction and correction.

        Parameters
        ----------
        detections : list
            Mirror boxes matched to this bike THIS frame, each
            `[conf, x1, y1, x2, y2]`. Usually 0, 1, or 2 items.
        movement_dx, movement_dy : float
            Legacy arguments retained for caller compatibility; unused.
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
        # The previous nearest-slot/center-shift implementation is replaced
        # by gated Kalman prediction and correction in _update_component_slots().
        self._update_component_slots(self.mirror_slots, detections, max_missed)
        if not self.mirror_seen_both:
            h0, h1 = (slot["hits"] for slot in self.mirror_slots)
            if h0 >= min_hits_for_both and h1 >= min_hits_for_both:
                self.mirror_seen_both = True

    def _update_component_slots(self, slots, detections, max_missed):
        """Predict component centers, gate detections, and correct matched slots."""
        gate = 9.21  # 99% chi-square gate for a two-dimensional measurement
        candidates = []

        # Predict each existing slot once for this frame.
        for si, slot in enumerate(slots):
            if slot["cord"] is None:
                continue
            if slot["kalman"] is None:
                c = slot["cord"]
                slot["kalman"] = ComponentKalman(((c[1] + c[3]) / 2, (c[2] + c[4]) / 2))
            slot["kalman"].predict()
            for di, det in enumerate(detections):
                center = ((det[1] + det[3]) / 2, (det[2] + det[4]) / 2)
                distance = slot["kalman"].measurement_distance(center)
                if distance <= gate:
                    candidates.append((distance, si, di))

        # One-to-one gated assignment. With two slots, sorting the valid
        # candidate costs is small and prevents distant detections taking
        # over a track.
        candidates.sort(key=lambda pair: pair[0])
        matched_slots, matched_dets = set(), set()
        for _, si, di in candidates:
            if si in matched_slots or di in matched_dets:
                continue
            matched_slots.add(si)
            matched_dets.add(di)
            det = detections[di]
            slot = slots[si]
            center = slot["kalman"].update(((det[1] + det[3]) / 2, (det[2] + det[4]) / 2))
            width, height = det[3] - det[1], det[4] - det[2]
            slot["cord"] = [
                det[0], center[0] - width / 2, center[1] - height / 2,
                center[0] + width / 2, center[1] + height / 2,
            ]
            slot["missed"] = 0
            slot["hits"] += 1

        # Unmatched tracks use their Kalman prediction briefly. Clear all
        # identity-specific state on timeout so later detections start fresh.
        for si, slot in enumerate(slots):
            if slot["cord"] is None or si in matched_slots:
                continue
            slot["missed"] += 1
            if slot["missed"] > max_missed:
                slot["cord"] = None
                slot["kalman"] = None
                slot["missed"] = 0
                slot["hits"] = 0
                slot["is_signal"] = False
                if "history" in slot:
                    slot["history"].clear()
                continue
            center = slot["kalman"].state[:2]
            width, height = slot["cord"][3] - slot["cord"][1], slot["cord"][4] - slot["cord"][2]
            slot["cord"] = [
                slot["cord"][0], center[0] - width / 2, center[1] - height / 2,
                center[0] + width / 2, center[1] + height / 2,
            ]

        # Start filters for unmatched detections in empty slots.
        empty_slots = [si for si, slot in enumerate(slots) if slot["cord"] is None]
        for di, det in enumerate(detections):
            if di in matched_dets or not empty_slots:
                continue
            si = empty_slots.pop(0)
            center = ((det[1] + det[3]) / 2, (det[2] + det[4]) / 2)
            slots[si]["cord"] = det
            slots[si]["kalman"] = ComponentKalman(center)
            slots[si]["missed"] = 0
            slots[si]["hits"] += 1
            slots[si]["is_signal"] = False
            if "history" in slots[si]:
                slots[si]["history"].clear()

    def update_indicators(self, detections, movement_dx=0.0, movement_dy=0.0,
                           max_missed=15, min_hits_for_both=3):
        """Update indicator slots using the same Kalman tracker as mirrors.

        Parameters
        ----------
        detections : list
            Indicator boxes matched to this bike THIS frame, each
            `[conf, x1, y1, x2, y2]`. Usually 0, 1, or 2 items.
        movement_dx, movement_dy : float
            Legacy arguments retained for caller compatibility; unused.
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
        # Legacy nearest-slot matching and bike-center translation are
        # inactive; both feature types now use the same Kalman slot tracker.
        self._update_component_slots(self.indicator_slots, detections, max_missed)
        if not self.indicator_seen_both:
            h0, h1 = (slot["hits"] for slot in self.indicator_slots)
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
        finalized for the frame (i.e. after the turn-zone/orientation
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
        turn_str = " TURNING (ZONE)" if self.is_turning else ""
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
    
    def update_box_color(self):
        """Apply the display color after turn and signal states are finalized.

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
