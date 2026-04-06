# FIRST Tech Challenge Automatic Scouter
| Internal Codename | **Project REDACTED** |
| -------- | --------|
| rayID Designation |RE-016236-08 |

*This project is developed by **Ray Enterprises' Advanced Research and Experimental Development Division** (ARED) for use by FTC Team Juice 16236*

## About AutoScout
Tired of manually scouting? Let the computer do it for you! Feed it a match video and it will track and log the paths each robot take, each action they perform, and even collect stats on where they are able to make shots.

Take the outputted WPILogs and bring them into AdvantageScope to rewatch a match in 3D! Or use the outputted JLOGs (Juice Logs) and create charts and detected strategy diagrams automaticly with our tools for your strategy team.

## Usage - Quickstart
### Step 1
Clone this repo and install the necessary packages/setup the environment using the `setup.py` file
### Step 2
Run the following command to run the scouting:
```shell
python3 autoscout.py <YOUTUBE_URL>
```
## Optional Flags
| Flag | Type | Default | Description |
| -------- | -------- | -------- | --------|
| `--output-dir` | `string` | `./output` | The folder that all outputted images, debug files, and logs will be stored |
| `--start-offset` | `float` | `0.0` | Seconds to skip before the match timer starts (Saves processing time) |
| `--sample-rate` | `int` | `10` | Frames per second to process |
| `--debug` | - | Not Enabled | Saves annotated debug frames to `tracker_debug/` |
| `--debug-every` | `int` | `5` | Save 1 debug frame every N processed frames |
| `--no-download` | - | Not Enabled | Use local video instead of YouTube video. **`--video-path` is required** |
| `--video-path` | `string` | `None` | Relative path to the locally stored match video. **`--no-download` is required** |
| `--corners` | `string` | `None` | Import custom calibrated field-borders from `calibrate.py `. Feed in `field_corners.json` unless specificly changed.

## Custom Field Calibration
`autoscout.py` will attempt to detect the field edges and create a boundary, but if you want to manually specify the boundaries, run the following command:
```bash
python3 calibrate.py <PATH_TO_LOCAL_FILE>
```
Please note the calibration app does not support YouTube videos.
When the app opens, follow the onscreen directions to generate a `field_corners.json` file. Please note, if you choose to go manual, you will need to use the `--corners` flag when running the scouting file, and you will need to recalibrate everytime the camera angle changes during the event

## Outputs
TBD I am still working on finalizing the Specs

## JLOG File Schema
JLOG is in a binary format that follows RULS (Ray Universal Log Schema). Please see below for the schema.

## Contributing
Please open a pull request with adequate information on the changes made. Keep PRs focused on a single issue (don't bundle multiple unrelated features/bug fixes into a single PR).