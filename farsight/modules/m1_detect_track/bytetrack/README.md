# bytetrack

ByteTrack via `ultralytics.trackers.byte_tracker.BYTETracker` (same two-stage high/low-score IoU + Kalman
association as ifzhang/ByteTrack). No weights. `update((N,5) boxes) -> (M,7) [x1,y1,x2,y2,id,score,det_idx]`
for confirmed tracks; `det_idx` links back to the detection (and its face).

Test: `bash run.sh pytest -q farsight/modules/m1_detect_track/bytetrack` -> 3 passed
(stable IDs for two synthetic walkers, determinism/empty input, low-score second-stage association).

Deviation from plan: plan lists the ifzhang/ByteTrack submodule; we wrap ultralytics' implementation instead
(already installed, maintained, same algorithm) - no re-implementation, no extra submodule.
`track_buffer` is scaled by fps/30 like ultralytics' track mode.
