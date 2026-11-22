# Mirror and Indicator Tracking Review

A constant-velocity Kalman filter is now used for mirror and indicator slot centers in `deep_sort/deep_sort/track.py`. The previous nearest-slot matching and bike-center translation path is inactive. These are the remaining limitations and possible next improvements.

## Kalman tracking now in place

- Each of the two mirror slots and two indicator slots tracks center position and velocity.
- The filter predicts the slot location each frame and corrects it when a detection passes a Mahalanobis-distance gate (`9.21`).
- A missed slot uses its predicted center for up to 15 frames, then clears. Clearing also resets hit count and indicator blink history.
- Bike bounding boxes continue to use DeepSORT's existing Kalman filter; this new filter applies to the smaller mirror and indicator boxes.
- The prior nearest-slot assignment and translation by bike-center motion are no longer active. Legacy movement arguments remain on the update methods for caller compatibility and are ignored.

## Remaining problems and possible solutions

### 1. Kalman noise and gate are uncalibrated

**Problem:** The component filter uses fixed process noise, measurement noise, initial covariance, and a `9.21` Mahalanobis gate. These values have not been tuned for the camera resolution, detector jitter, or component size. As covariance grows during misses, the gate can accept detections farther away in pixels.

**Possible solution:** Tune the noise and gate on labeled clips. Consider adding a second hard displacement limit normalized by bike-box size. Log rejected and accepted distances during a short calibration run.

### 2. Gated pair assignment is still greedy

**Problem:** Valid slot/detection pairs are sorted by cost and assigned one by one. In ambiguous frames, that local choice may not yield the best overall pairing.

**Possible solution:** Use a minimum-cost one-to-one assignment, such as the Hungarian algorithm. Since there are at most two slots per component type, comparing the possible assignments directly is also simple.

### 3. “Both seen” still relies on lifetime hit counts

**Problem:** Each slot's hit count accumulates across the track. `mirror_seen_both` and `indicator_seen_both` can become true even if the two slots were detected at different times, and the flags remain true for the rest of the track.

**Possible solution:** Track hit timestamps or a short rolling window. Require recent hits for both slots, preferably including several frames where both are detected together, before setting the summary flag.

### 4. Predicted boxes keep their last observed size

**Problem:** The Kalman filter estimates center position and velocity only. During misses, box width and height remain at their last detected values, which may be inaccurate as the bike changes scale or orientation.

**Possible solution:** Extend the state to estimate width and height, or store component geometry relative to the bike box and reconstruct it from the current bike box. Clear predictions that leave the bike box or frame.

### 5. Component-to-bike association can still choose the wrong bike

**Problem:** Before slot tracking, `best_match_track()` assigns each mirror/indicator detection to a bike using center containment and IoU. Overlapping bike boxes can make this assignment ambiguous; a slot-level Kalman filter cannot correct a component assigned to the wrong bike.

**Possible solution:** Score candidate bikes with normalized component-center distance, containment, and IoU. Add an ambiguity margin and leave a detection unassigned when two bikes score similarly. Reuse the previous bike association when the evidence remains plausible.

### 6. Slot indices do not mean left and right

**Problem:** Empty slots are initialized in detector output order, so slot 0 and slot 1 are not stable semantic left/right labels. The filter maintains a slot identity, but it does not establish which side of the motorcycle that slot represents.

**Possible solution:** If downstream behavior needs the active signal side, derive semantic side from bike orientation and feature position relative to the bike. Keep this label separate from the filter's internal slot ID. If the app only needs “any indicator is on,” document that slot indices have no left/right meaning.

### 7. Blink inference repeats work on the full image history

**Problem:** On turning frames, each available indicator slot gets a new crop and the blink model preprocesses and evaluates its stored sequence. Repeating this for many bikes can add substantial per-frame cost.

**Possible solution:** Run inference at a configurable interval or after a minimum number of new samples. Use a fixed-length rolling window, cache preprocessed samples, and smooth predictions over several results. Compare throughput and signal accuracy on the same clips.

### 8. Blink history can include predicted or poor crops

**Problem:** Blink crops are taken from slot coordinates even when the current location came from Kalman prediction rather than a fresh detector result. The crop may no longer contain the indicator, but it still enters the sequence.

**Possible solution:** Mark whether each slot location came from a real detection or prediction. Prefer fresh, high-confidence crops for blink classification; skip or down-weight predicted crops. Reset history when a slot is reassigned or times out.

### 9. Blink confidence does not stabilize the signal state

**Problem:** The blink model returns a confidence score, but the pipeline sets the slot state from the predicted class directly. A single uncertain result can toggle the signal state.

**Possible solution:** Use a confidence threshold and short temporal smoothing, such as a majority vote or hysteresis. Tune the thresholds on labeled on/off indicator sequences and retain raw predictions for debugging.

## Suggested next steps

1. Measure current slot-tracking accuracy and runtime on a small labeled video set.
2. Calibrate the Kalman noise/gate and add a bike-size displacement cap if needed.
3. Improve ambiguous component-to-bike association and require recent joint evidence for “both seen.”
4. Reduce repeated blink inference and exclude low-quality predicted crops.
5. Re-run the same clips and compare ID switches, missed detections, false slot matches, blink accuracy, and processing time.
