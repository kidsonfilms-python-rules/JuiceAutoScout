![Project Header Image](assets/Project-REDACTED-Header.png)

# *FIRST* Tech Challenge Automatic Scouter
| Internal Codename | **Project REDACTED** |
| -------- | -------- |
| rayID Designation | RE-016236-08 |

*This project is developed by **Ray Enterprises' Advanced Research and Experimental Development Division** (ARED) for use by FTC Team Juice 16236.*

## About AutoScout
`auto_scout.py` tracks the four robots in an FTC match video, projects them into field coordinates, and exports the result for later analysis.

Current outputs are focused on robot motion:
- `robot_positions.csv` for per-timestamp robot poses and visibility
- `match_log.wpilog` for AdvantageScope playback
- optional annotated debug frames in `output/tracker_debug/`

Output coordinates now use a center-origin field frame:

- `(0, 0)` is the center of the 144 inch x 144 inch field
- `x` increases to the right
- `y` increases downward in the projected field plane used by the tracker
- the field corners are approximately `(-72, -72)` to `(72, 72)` in inches

## Requirements
Install the runtime dependencies:

```bash
pip install opencv-python numpy progress
pip install scipy
```

`scipy` is technically optional, but the tracker uses it for the best blob-to-track assignment.

## Quickstart
Track a local video:

```bash
python3 auto_scout.py --no-download --video-path match.mp4 --corners field_corners.json
```

Track from YouTube:

```bash
python3 auto_scout.py "https://www.youtube.com/watch?v=..."
```

## CLI Flags
| Flag | Type | Default | Description |
| -------- | -------- | -------- | -------- |
| `--output-dir` | `string` | `./output` | Directory for CSV, WPILOG, background image, and debug frames |
| `--start-offset` | `float` | `0.0` | Seconds to skip before the match timer starts |
| `--sample-rate` | `float` | `10.0` | Effective frames per second to process |
| `--debug` | - | disabled | Save annotated debug frames to `tracker_debug/` |
| `--debug-every` | `int` | `5` | Save one debug frame about every `N` source-video frames |
| `--no-download` | - | disabled | Use a local video instead of downloading from YouTube |
| `--video-path` | `string` | `None` | Path to the local match video. Requires `--no-download` |
| `--corners` | `string` | `None` | Path to `field_corners.json` from `calibrate.py` |
| `--robot-init-positions` | `json` | `None` | Four starting field coordinates in inches, using center-origin coordinates |
| `--manual-reference-csv` | `string` | `None` | Manual `robot_positions`-style CSV in the same center-origin coordinate system |

## Field Calibration
AutoScout can try to detect the field outline automatically, but manual calibration is more reliable.

Generate field corners from a local video:

```bash
python3 calibrate.py path/to/match.mp4
```

That produces a `field_corners.json` file. Pass it to `auto_scout.py` with `--corners`. Recalibrate whenever the camera angle or crop changes.

## Outputs
When a run finishes, the output directory typically contains:

- `robot_positions.csv`
  Flat table of timestamped robot positions, headings, and visibility flags. `x/y` are in inches from field center.
- `match_log.wpilog`
  WPILOG output using the same center-origin coordinates, converted to meters.
- `median_background.jpg`
  The median background image used for subtraction.
- `tracker_debug/`
  Optional annotated frames showing contours, split centers, merge annotations, and robot IDs.

## Debug Notes
- `--debug-every` is based on source-video frame spacing, not processed-frame count.
- Merge debug behavior can also be influenced by the top-level `SAVE_ALL_DEBUG_AROUND_MERGES` flag in `auto_scout.py`.
- Each debug run clears old `.jpg` files from `output/tracker_debug/` before writing new ones.

## AdvantageScope
To inspect the resulting WPILOG:

1. Open AdvantageScope.
2. Open `match_log.wpilog`.
3. Add a `2D Field` tab.
4. Set the field to the FTC season field you want.
5. Drag each `Robot#/Pose` entry into the field poses list.

## Contributing
Please open a focused pull request with a clear explanation of the change. Avoid bundling unrelated fixes into a single PR.

## License
This project is licensed under the MIT license, please see `LICENSE` for more information.
