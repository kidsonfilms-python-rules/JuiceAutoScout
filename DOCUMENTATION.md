# JuiceAutoScout Documentation

## Overview

This repository tracks the four robots in an FTC match video, projects them into field coordinates, and exports the result for analysis and visualization. The main workflow is:

1. Calibrate the field corners in the video.
2. Run the automatic tracker on the calibrated video.
3. Inspect the CSV, WPILOG, background image, and optional debug frames.
4. If needed, produce supervised manual labels with the browser-based manual tracker and reuse them to tune identity assignment.

The repo currently contains four user-facing tools:

- `auto_scout.py`: main automatic tracker.
- `calibrate.py`: interactive corner picker that produces `field_corners.json`.
- `manual_tracker.html`: browser UI for hand-labeling robot poses and exporting CSV/JSON.
- `debug.py`: small utility that prints the structure of a generated WPILOG.

## Repository Layout

- `auto_scout.py`: full tracking pipeline, exports, and CLI.
- `calibrate.py`: manual field calibration tool.
- `manual_tracker.html`: no-build manual annotation app.
- `debug.py`: WPILOG inspection helper.
- `field_corners.json`: example/manual calibration file.
- `output/`, `output_mergefix/`: example output directories.
- `manual_robot_positions.csv`: example hand-labeled robot CSV.
- `README.md`: short quickstart; `DOCUMENTATION.md` is the full reference.

## Dependencies

### Python runtime

Required in practice:

```bash
pip install opencv-python numpy
```

Recommended:

```bash
pip install scipy progress
```

What each dependency is used for:

- `opencv-python`: video I/O, homographies, contours, morphology, color masks.
- `numpy`: matrix math and array operations.
- `scipy`: Hungarian assignment via `scipy.optimize.linear_sum_assignment`.
- `progress`: nicer terminal progress bars.

Optional/external:

- `yt-dlp`: required only when downloading a video from YouTube instead of using `--no-download --video-path`.

The code has fallbacks for missing `progress` and `scipy`, but tracking quality is best with `scipy` installed.

## Core Concepts

### Field coordinate system

The public coordinate system is now center-origin on a normalized 144 x 144 inch FTC field.

- `(0, 0)` is the center of the field.
- `x` increases from left to right.
- `y` increases from top to bottom in the transformed field plane used by the code.
- the corners are `(-72, -72)`, `(72, -72)`, `(72, 72)`, and `(-72, 72)` in inches
- positions are stored in inches in CSV output
- WPILOG output converts those centered inches to meters using `in * 0.0254`

Implementation detail:

- the tracker still keeps its internal homography and state on the historical `0..144` field plane
- external interfaces and outputs are converted by subtracting `72` inches from both axes

### Robot IDs

There are always four logical tracks: `Robot0` through `Robot3`.

On automatic initialization, IDs are assigned left-to-right using the best visible four-blob lineup near the start of the match. That means the IDs are not alliance-aware by default; they are just consistent logical labels.

### Visibility

Each pose row includes a visibility flag.

- `visible=1`: tracker believes the robot is currently seen or can still be safely coasted.
- `visible=0`: the track has dropped out.

When a robot temporarily disappears, the tracker can keep its last position alive for a short coast window.

## Typical Workflows

### Workflow 1: Track a local video

1. Generate field corners:

```bash
python3 calibrate.py match.mp4
```

2. Track the video:

```bash
python3 auto_scout.py --no-download --video-path match.mp4 --corners field_corners.json
```

3. Open the outputs in `./output` by default.

### Workflow 2: Track a YouTube match

```bash
python3 auto_scout.py "https://www.youtube.com/watch?v=..."
```

This path requires `yt-dlp`. If no corners are provided, the tracker will attempt automatic field detection.

### Workflow 3: Use manual initialization for bad starting conditions

If robots start under field structures or are otherwise hard to see at match start, pass explicit starting field coordinates:

```bash
python3 auto_scout.py \
  --no-download \
  --video-path match.mp4 \
  --corners field_corners.json \
  --robot-init-positions '[[-52.7,-70.4],[-12.6,53.0],[15.8,60.6],[57.5,-55.7]]'
```

This bypasses the normal blob-based bootstrap step.

### Workflow 4: Build supervised re-ID references

1. Label a match manually with `manual_tracker.html`.
2. Export `manual_robot_positions.csv`.
3. Re-run the tracker with:

```bash
python3 auto_scout.py \
  --no-download \
  --video-path match.mp4 \
  --corners field_corners.json \
  --manual-reference-csv manual_robot_positions.csv
```

This uses the manual labels to build per-robot appearance histograms that help identity assignment.

## `auto_scout.py`

## CLI

```bash
python3 auto_scout.py [url]
```

Flags:

- `--output-dir`: output directory. Default `./output`.
- `--start-offset`: seconds skipped before the match timer starts. Default `0.0`.
- `--sample-rate`: intended processed FPS. Default `10.0`.
- `--debug`: saves annotated debug frames to `tracker_debug/`.
- `--debug-every`: save one debug frame every N source frames. Default `5`.
- `--debug-enable-hitboxes`: draws wireframe robot hitboxes in debug frames.
- `--no-download`: use a local video instead of YouTube.
- `--video-path`: local video path. Requires `--no-download`.
- `--corners`: path to `field_corners.json`.
- `--robot-init-positions`: JSON file path or inline JSON string of four center-origin `[x, y]` field positions in inches.
- `--manual-reference-csv`: manual `robot_positions`-style CSV used to build supervised appearance references.

### Important behavior note about `--sample-rate`

The file currently sets:

```python
PROCESS_EVERY_SOURCE_FRAME = True
```

Because of that, `process_match()` forces `frame_step = 1`, so the tracker currently processes every source frame regardless of `--sample-rate`. The code still prints the nominal output rate, but in the current version `--sample-rate` is effectively overridden unless `PROCESS_EVERY_SOURCE_FRAME` is changed.

## Outputs

Each run writes:

- `robot_positions.csv`: timestamped field poses and visibility flags.
- `match_log.wpilog`: AdvantageScope-compatible output.
- `median_background.jpg`: median background used for subtraction.
- `tracker_debug/`: optional annotated frames when `--debug` is enabled.

### CSV schema

Header:

```csv
timestamp_s,
robot0_x_in,robot0_y_in,robot0_heading_rad,robot0_visible,
robot1_x_in,robot1_y_in,robot1_heading_rad,robot1_visible,
robot2_x_in,robot2_y_in,robot2_heading_rad,robot2_visible,
robot3_x_in,robot3_y_in,robot3_heading_rad,robot3_visible
```

Notes:

- `timestamp_s` is match time, not raw video timestamp.
- It is computed as `current_frame / video_fps - start_offset`.
- `robotN_x_in` and `robotN_y_in` are center-origin coordinates in inches.
- When a robot is missing, the row uses zeros and `visible=0` if the track is not being coasted.

### WPILOG contents

The writer creates:

- `/Robot0/Pose` through `/Robot3/Pose`
- `/Robot0/Visible` through `/Robot3/Visible`

Pose entries are written as `x_m`, `y_m`, and `heading_rad`.

Important note:

- the WPILOG coordinates are also center-origin now
- if you use a viewer that assumes a corner-origin field frame, the poses will appear offset unless that viewer is configured for a center-origin convention

## Processing Pipeline

### 1. Video open and timing setup

The tracker opens the video with OpenCV and reads:

- source FPS
- frame count
- frame size

It computes:

- `start_frame = start_offset_sec * video_fps`
- `match_time_s = frame_num / video_fps - start_offset_sec`
- `timestamp_us = int(match_time_s * 1_000_000)`

### 2. Field calibration and homography

If `--corners` is supplied, the tracker loads:

```json
{"corners_px": [[bl_x, bl_y], [br_x, br_y], [tr_x, tr_y], [tl_x, tl_y]]}
```

It reorders those into `[tl, tr, br, bl]`, then builds a homography from image pixels to a square field:

- image corners: `[tl, tr, br, bl]`
- field corners: `[(0,0), (144,0), (144,144), (0,144)]`

This yields:

- `H_2d`: image pixel -> field inches
- `H_inv`: field inches -> image pixel

Internally, those field coordinates still use the legacy corner-origin plane:

- top-left: `(0, 0)`
- top-right: `(144, 0)`
- bottom-right: `(144, 144)`
- bottom-left: `(0, 144)`

The center-origin public coordinates are derived afterward by subtracting `(72, 72)`.

If no corners are supplied, `FieldDetector.detect_field()` tries to find a large quadrilateral automatically using:

- grayscale conversion
- Gaussian blur
- Canny edges
- contour extraction
- polygon approximation (`approxPolyDP`)

This is a fallback, not the preferred workflow.

### 3. Background model

`RobotTracker.setup()` builds a static background image by sampling `N_BG_SAMPLES = 80` frames uniformly across the whole video and taking the per-pixel median.

Why median:

- moving robots disappear from the estimate if they are not in the same pixel most of the time
- lighting and camera noise are suppressed better than with a single frame

This background is saved as `median_background.jpg`.

### 4. Field mask

The tracker fills a polygon slightly padded around the detected field. This removes most audience, scoreboard, and off-field clutter.

The current mask padding is asymmetric:

- side pad: `10`
- top pad: `12`
- bottom pad: `18`

That is a hand-tuned compromise:

- tighter on the sides to reduce bleed from people
- a little extra room at the far wall
- reduced bottom padding to trim scoreboard noise

### 5. Robot size estimation in pixels

The tracker projects the field center and points offset by `ROBOT_SIZE_IN = 18` inches in the `x` and `y` directions back into image space. From those projected distances it estimates how many pixels an 18-inch robot spans in the current camera geometry.

This value drives a lot of later thresholds:

- merge splitting
- reassociation windows
- appearance crop size
- wireframe debug hitboxes

### 6. Foreground extraction

Foreground is computed from the current frame and the background:

1. `absdiff(frame, background)`
2. grayscale conversion
3. binary threshold with `FG_THRESH = 30`
4. field masking
5. morphological close
6. morphological open
7. optional "neck-breaking" erode/dilate using a kernel scaled from robot size

The neck-breaking step is specifically there to separate thin bridges in merged blobs.

### 7. Ball removal

FTC game pieces can look like small robot blobs, so the tracker removes color regions matching configured HSV bands:

- green balls
- purple balls

Ball masks are:

- thresholded in HSV
- opened and closed with size-scaled kernels
- filtered by projected field area
- dilated slightly before subtraction

The final foreground mask subtracts this ball mask.

### 8. Contour extraction and field-area filtering

From the cleaned foreground mask, the tracker extracts contours and filters them by:

- pixel area: `BLOB_MIN = 350`
- projected field area: `BLOB_MIN_FIELD_AREA_IN2 = 40.0`
- minimum distance-transform radius: `MIN_RADIUS_PX = 6`

Projected field area matters because perspective makes far-away robots smaller in pixels. A contour that is tiny in image space can still be a real robot if it maps to a plausible field footprint.

### 9. Blob splitting for merged robots

Each contour is treated as containing one or more robots. The tracker estimates robot centers inside a contour using a distance transform:

1. Fill the contour into a binary mask.
2. Run `distanceTransform`.
3. Threshold peaks at `0.40 * dist.max()`.
4. Erode with a separation kernel.
5. Use connected components and component centroids as candidate robot centers.

The tracker estimates how many robots a contour probably contains from area ratio:

- 1 robot: area ratio below `1.6`
- 2 robots: ratio at least `1.6`
- 3 robots: ratio at least `2.6`
- 4 robots: ratio at least `3.5`

If the strict split underestimates the expected count, it retries with relaxed thresholds:

- multi-robot relaxed peak ratio: `0.30`
- pair relaxed peak ratio: `0.35`
- multi-robot separation kernel: `3`
- pair separation kernel: `5`

If that still misses robots, `_augment_subcenters_with_track_priors()` can inject predicted track positions into the candidate center list for under-resolved merge blobs.

### 10. Candidate blob records

For every split center, the tracker creates a candidate blob entry containing:

- field `x`, `y`
- parent contour ID
- image centroid `cx`, `cy`
- quality penalty
- split count inside the parent contour
- contour handle
- optional appearance feature histogram

Candidates are sorted by split-adjusted area so larger, clearer detections are preferred earlier.

### 11. Static blob suppression

Persistent non-robot foreground clutter can trap tracks. The tracker maintains a coarse history of blob centroids and suppresses candidates that:

- remain within `STATIC_BLOB_MOVE_PX = 8` pixels
- for at least `STATIC_BLOB_SUPPRESS_FRAMES = 20` consecutive frames
- unless a currently tracked robot is already very near that blob

This is aimed at stationary field structures and other static artifacts.

### 12. Corner-zone suppression

Top corner field elements can create persistent false blobs. The tracker suppresses detections in two 22-inch top-corner squares unless a currently active robot is already nearby.

This reduces the common failure mode where a track gets pulled into a corner post.

### 13. Track initialization

Until the tracker is initialized, it waits for at least four candidate blobs, then chooses the best four-blob lineup from up to `INIT_MAX_CANDIDATES = 8` candidates.

Scoring:

- blobs are sorted left-to-right in field space
- the tracker penalizes too-small spacing between adjacent robots
- it also adds each blob's quality penalty

The chosen lineup becomes `Robot0` through `Robot3`.

If you pass `--robot-init-positions`, initialization is forced immediately from the provided center-origin field coordinates instead.

### 14. Motion model

Each track stores:

- field position `_pos`
- image position `_pos_px`
- smoothed field velocity `_vel`
- smoothed image velocity `_vel_px`
- coast counter `_coast`

Prediction is simple linear extrapolation:

- `predicted_pos = pos + vel * horizon`
- `horizon = min(1.0 + 0.15 * coast, 2.0)`

Velocity smoothing is:

- `new_vel = 0.45 * previous_vel + 0.55 * raw_delta`

Field velocity is clipped to `MAX_SPEED_IN = 140.0` per processed frame before smoothing.

Heading is derived from motion, not shape:

- if speed magnitude is below `0.25`, keep the previous heading
- otherwise `heading = atan2(vy, vx)`

### 15. Assignment logic

The tracker assigns up to four logical tracks to the current candidate blobs using a cost matrix.

Base components of the cost:

- field distance from predicted position
- image distance from predicted pixel position
- blob quality penalty
- appearance mismatch

Extra penalties apply after merge resolution if a temporary post-merge lock is active.

The real cost is:

- `dist_cost`
- `+ 0.35 * img_cost`
- `+ qual_cost`
- `+ appearance_weight * appearance_cost`
- `+ post_merge_lock penalties when active`

Matching is also gated by hard plausibility rules:

- reject if field distance is beyond reacquisition distance
- reject if image distance exceeds a robot-size-scaled limit
- reject weak-looking single-split matches during certain coast/merge states

The assignment solver:

- preferred: Hungarian assignment from SciPy
- fallback: greedy minimum-cost matching if SciPy is unavailable

Skip columns are added to the cost matrix so a track is allowed to remain unmatched when all real matches are bad.

### 16. Appearance re-identification

Each candidate can carry an appearance descriptor:

- HSV histogram
- hue bins: `18`
- saturation bins: `16`
- extracted from a circular crop around the candidate center
- optionally masked by the contour itself

Appearance distance uses Bhattacharyya distance:

- `cv2.compareHist(..., HISTCMP_BHATTACHARYYA)`

If `--manual-reference-csv` is provided, the tracker samples frames from the labeled CSV and builds up to `REID_MAX_SAMPLES_PER_ROBOT = 48` reference histograms per robot, using every `REID_SAMPLE_STRIDE = 15`th row.

These references make appearance cost meaningful for identity assignment and especially reacquisition.

Important caveat:

- the current loader converts `timestamp_s` directly into `frame_num = timestamp_s * video_fps`
- it does not add `start_offset`
- the CSV coordinates are interpreted as center-origin and converted back to the internal `0..144` field plane before projection
- in practice, manual reference CSVs should be exported against the raw video timeline starting at `0`, or the sampled frames will be shifted early

### 17. Merge detection

After assignment, the tracker groups assigned tracks by parent contour ID. If two or more tracks map into the same parent contour, that contour is treated as an active merge.

The merge state lives in `MergeGroup`, which stores:

- `track_ids`
- `entry_axis`
- `parent_id`
- `crossed` for 2-robot merges
- `entry_order` for 3+ robot merges
- `current_order`
- `peak_assignment`
- `order_votes`
- `entry_features`

### 18. Two-robot merges

For a 2-robot merge:

- the tracker computes the merge entry axis from the farthest-apart pair of track positions
- inside the merge, it checks whether the split peaks have crossed along that axis
- on separation, it swaps track state if `crossed=True`

This preserves identity through a simple side-swap.

### 19. Three- and four-robot merges

This repo's current tracker version has explicit fixes for 3+ robot merges.

The important ideas are:

- while robots are merged, each track is anchored to its current peak, not frozen at merge entry
- the tracker records the order of robots along the entry axis at merge start
- each frame it updates the current order from split peaks
- partial or stale split frames do not erase the last good peak assignment
- on separation, the tracker applies the permutation implied by `entry_order -> current_order`

The current implementation also keeps weighted order votes over time and can relabel directly from exit blobs when the split frame gives good candidates.

That is the core logic that fixes long multi-robot merge cases where robots rearrange while overlapped.

### 20. Post-merge locks

After a merge resolves, the tracker can keep a short-lived lock for each robot containing:

- image position and velocity
- field position and velocity
- last appearance feature
- merge group key

These locks:

- bias assignment toward the correct continuation
- reject implausibly distant jump assignments
- help prune duplicate post-merge assignments

This is an additional layer on top of the merge permutation logic.

### 21. Debug overlays

When `--debug` is enabled, saved frames may show:

- detected field polygon
- foreground contours
- split centers
- ball-mask contours
- robot centers and ID labels
- velocity arrows
- merge arrows and labels
- optional wireframe robot hitboxes

Merge labels mean:

- 2-robot merge: `ok` or `CROSSED`
- 3+ robot merge: `[entry_order→current_order]`

Example:

- `[012→201]` means the left-to-right order at merge entry was `0,1,2`, and the current left-to-right peak order is `2,0,1`.

### 22. Wireframe hitbox debug mode

`--debug-enable-hitboxes` calls `_draw_wire_cube()`, which tries to approximate each robot as a wireframe box.

How it works:

1. Find the connected foreground component nearest the tracked center.
2. Use Canny edges on the component.
3. Estimate the floor contact point from the lower silhouette.
4. Back-project that point to field coordinates.
5. Binary-search a square base width in inches whose projected footprint matches the observed silhouette width.
6. Draw a projected cube with estimated height.

This is a visualization tool, not part of the assignment logic.

## `calibrate.py`

`calibrate.py` is the preferred way to generate `field_corners.json`.

Usage:

```bash
python3 calibrate.py match.mp4
python3 calibrate.py match.mp4 --output my_corners.json
python3 calibrate.py match.mp4 --frame 900
```

What it does:

- opens a chosen frame, defaulting to the middle frame of the video
- displays four draggable corners
- lets you zoom and pan
- writes `{"corners_px": [bl, br, tr, tl]}` on save

Controls:

- left-drag near a point: move that corner
- mouse wheel: zoom
- right-drag or middle-drag: pan
- `R`: reset corners
- `+` / `-`: zoom
- `Enter` or `S`: save
- `Esc` or `Q`: quit without saving

Internal details:

- the UI starts with default corners at 20% and 80% of frame width/height
- the editor shows corners as `TL`, `TR`, `BR`, `BL`
- on save it converts from display order `[TL, TR, BR, BL]` to tracker order `[BL, BR, TR, TL]`

## `manual_tracker.html`

This file is a standalone browser-based manual annotation tool. It is meant for creating ground-truth robot pose labels, especially when:

- the automatic tracker struggles
- you need supervised re-ID references
- you want a gold-standard CSV for comparison

### How to use it

1. Open `manual_tracker.html` in a browser.
2. Load a video file.
3. Load `field_corners.json`.
4. Select a robot (`R0` to `R3`).
5. Click the robot location on the video, or edit pose values directly.
6. Mark frames across the timeline.
7. Export CSV or session JSON.

No server or build step is required.

### Features

- load local video
- load field corners JSON
- load saved session JSON
- per-robot label timelines
- visible/hidden flag per label
- direct editing of center-origin `x`, `y`, and heading
- click-to-label on the video
- frame stepping
- interpolation between labels
- CSV export in the same schema as `auto_scout.py`
- JSON export of the full editable session

### Keyboard controls

- `1` to `4`: select robot
- `Space`: play/pause
- `Left` / `Right`: step video by one unit
- `Shift+Left` / `Shift+Right`: step by five units
- `Enter`: mark current pose
- `Backspace` or `Delete`: delete nearest label
- `V`: toggle visible flag

### Export behavior

CSV export settings:

- `FPS`: export sample rate
- `Start`: start time in seconds
- `End`: end time in seconds
- `Interpolate`: whether unlabeled timestamps between labels are interpolated

Interpolation logic:

- `x` and `y`: linear interpolation
- `heading`: shortest-path angular interpolation
- `visible`: inherited from the nearer side of the span

If interpolation is disabled, only exact labels are exported for a robot; otherwise missing rows for that robot become zeros.

The manual tracker now uses the same center-origin convention as `auto_scout.py` exports:

- field center is `(0, 0)`
- valid user-facing `x/y` values are approximately `[-72, 72]`

### Session JSON format

The exported JSON stores:

- metadata
- video name/duration/fps
- field corners
- export settings
- label timelines for each robot

This is useful for revising a labeling session later without losing the editable keyframes.

### Manual tracker math

The browser tool computes homography itself in plain JavaScript.

It solves an 8x8 linear system for a 3x3 projective transform with `h33 = 1`, then inverts the matrix to support both:

- image -> field (`imageToField`)
- field -> image (`fieldToImage`)

Like the Python tracker, the browser tool keeps the homography on the historical `0..144` field plane internally, then converts to or from center-origin coordinates by shifting each axis by `72` inches.

## `debug.py`

`debug.py` is a small inspection helper for generated WPILOG files.

It:

- reads `./output/match_log.wpilog`
- verifies the header
- parses entry-start records
- prints the first few data records in decoded form

This is useful when validating whether the logger is writing the expected pose and visibility channels.

## File Formats

## `field_corners.json`

Expected structure:

```json
{
  "corners_px": [
    [bl_x, bl_y],
    [br_x, br_y],
    [tr_x, tr_y],
    [tl_x, tl_y]
  ]
}
```

This ordering matters. `auto_scout.py` assumes this exact corner order when it rebuilds the homography.

## `robot-init-positions`

Accepted as:

- inline JSON string
- path to a JSON file

Expected content:

```json
[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
```

The values are field inches, already sorted by the logical robot ID you want to assign.

These values are center-origin:

- center of field: `(0, 0)`
- top-left: `(-72, -72)`
- bottom-right: `(72, 72)`

## Manual reference CSV

The tracker expects the same schema as `robot_positions.csv`.

It reads:

- `timestamp_s`
- `robotN_x_in`
- `robotN_y_in`
- `robotN_visible`

It uses those rows to jump back into the source video, project the labeled field position into image space, and sample appearance histograms around those points.

Those `robotN_x_in` / `robotN_y_in` values are center-origin coordinates in inches.

Because the current implementation treats `timestamp_s` as raw video time, not match-relative time, be careful when reusing CSVs produced with a nonzero `--start-offset` or a nonzero manual tracker export start.

## Tuning Constants and What They Mean

These are the important tracker constants in the current code:

- `FG_THRESH = 30`: background subtraction threshold.
- `BLOB_MIN = 350`: minimum contour area in pixels.
- `BLOB_MIN_FIELD_AREA_IN2 = 40.0`: minimum projected contour area on the field.
- `MIN_RADIUS_PX = 6`: minimum contour thickness from distance transform.
- `MAX_COAST = 60`: maximum coast frames remembered internally.
- `VISIBLE_COAST = 8`: how long a missing track can still be considered visible.
- `MAX_DIST_IN = 30.0`: normal field-space match distance.
- `MAX_REACQ_IN = 144.0`: larger distance allowed for reacquisition.
- `MAX_SPEED_IN = 140.0`: cap on per-frame field delta before smoothing.
- `MERGE_HOLD = 16`: merge memory window for assignment gating.
- `REID_COST_WEIGHT = 1.60`: normal appearance cost weight.
- `REID_COST_WEIGHT_REACQ = 2.70`: stronger appearance weight during reacquisition.
- `POST_MERGE_LOCK_FRAMES = 12`: lock duration after merge relabeling.

If you change camera angle or resolution substantially, the constants most likely to matter first are:

- field corners
- foreground threshold
- blob minimum area
- minimum radius
- merge split thresholds

## Assumptions and Limitations

- The tracker assumes a mostly static camera.
- Good field corners matter a lot. Bad calibration poisons everything downstream.
- The field is modeled as a flat plane. Any height above the field introduces projection error.
- Robot heading is motion-derived, not pose-estimated from chassis orientation.
- Automatic field detection is a convenience fallback, not the robust path.
- The tracker is heavily tuned for *FIRST* Championships-style overhead or elevated side-angle footage with four robots on one field.
- `--sample-rate` is currently bypassed by `PROCESS_EVERY_SOURCE_FRAME = True`.
- Appearance re-ID is optional and only becomes meaningful when you provide manual reference data.

## Failure Modes and Recovery

If the tracker initializes the wrong four robots:

- use `--robot-init-positions`
- tighten field corners
- inspect debug frames for false blobs

If robots disappear into corners:

- verify corner calibration
- use manual initialization
- inspect whether corner-zone suppression is too aggressive or too weak for the footage

If IDs swap after contact:

- enable debug frames and inspect merge labels
- provide `--manual-reference-csv`
- compare the split peaks and post-merge relabel behavior

If the tracker follows game pieces:

- inspect the ball mask in debug output
- adjust ball HSV thresholds in code if the event lighting differs significantly

If the output timing looks wrong:

- check `--start-offset`
- remember `timestamp_s` is relative to match start, not the raw file start

## AdvantageScope Use

To visualize the WPILOG:

1. Open AdvantageScope.
2. Open `match_log.wpilog`.
3. Add a `2D Field` tab.
4. Set the field to the FTC season field you want.
5. Drag `Robot0/Pose` through `Robot3/Pose` into the pose list.
6. Ensure the display expects meters and radians.

The WPILOG writer already converts centered inches to meters before logging.

If your viewer assumes a corner-origin field, you may need to mentally account for the center-origin shift or configure the visualization accordingly.

## Practical Recommendations

- Use `calibrate.py` for every materially different camera angle or crop.
- Prefer local video plus `--corners` over relying on automatic field detection.
- Turn on `--debug` whenever you are tuning thresholds or investigating identity swaps.
- Use `manual_tracker.html` when you need truth data instead of guessing from tracker output.
- Install `scipy`; the Hungarian assignment is better than the greedy fallback.

## Summary

At a high level, this project is a perspective-aware multi-object tracker specialized for FTC match footage. Its main ingredients are:

- a calibrated projective transform from image space to a 144-inch field plane
- median-background foreground subtraction
- contour splitting with distance-transform peaks
- four persistent logical tracks with motion prediction
- appearance-assisted assignment
- explicit identity-preserving merge logic, including permutation tracking for 3+ robot overlaps
- export to CSV and WPILOG for downstream analysis

If you understand those pieces, you understand most of how the repository works.
