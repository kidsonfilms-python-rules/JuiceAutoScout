# -*- coding: utf-8 -*-
import argparse
import csv
import json
import math
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Lazy imports
# ─────────────────────────────────────────────
def _require(package, pip_name=None):
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        name = pip_name or package
        print("[ERROR] Missing dependency: {}\n  Install with:  pip install {}".format(name, name))
        sys.exit(1)

# ─────────────────────────────────────────────
# WPILOG Writer (Strict Spec v1.0)
# ─────────────────────────────────────────────
class WPILogWriter:
    POSE2D_SCHEMA = "double x;double y;struct:Rotation2d rotation"
    ROTATION2D_SCHEMA = "double value"
    HEADER_MAGIC = b"WPILOG"
    def __init__(self, path):
        self._fh = open(path, "wb")
        self._next_id = 1
        self._write_file_header()
    def _write_file_header(self):
        # Spec requires 12 bytes: Magic(6), Ver(2), Extra(4)
        self._fh.write(self.HEADER_MAGIC)
        self._fh.write(struct.pack("<BBI", 1, 0, 0)) 
    def _encode_int(self, value, max_bytes):
        for n in [1, 2, 4, 8]:
            if n > max_bytes: break
            if value < (1 << (8 * n)):
                return value.to_bytes(n, "little"), n
        return value.to_bytes(max_bytes, "little"), max_bytes
    def _write_record(self, entry_id, timestamp_us, data):
        eid_bytes, eid_len = self._encode_int(entry_id, 4)
        size_bytes, size_len = self._encode_int(len(data), 4)
        ts_bytes, ts_len = self._encode_int(timestamp_us, 8)
        bitfield = ((eid_len - 1) & 0x03) | (((size_len - 1) & 0x03) << 2) | (((ts_len - 1) & 0x07) << 4)
        self._fh.write(struct.pack("<B", bitfield))
        self._fh.write(eid_bytes)
        self._fh.write(size_bytes)
        self._fh.write(ts_bytes)
        self._fh.write(data)
    def _register_schema(self, type_name, schema_str):
        name = "/.schema/" + type_name
        payload = struct.pack("<BI", 0, self._next_id) + struct.pack("<I", len(name)) + name.encode("utf-8") + struct.pack("<I", 12) + b"structschema" + struct.pack("<I", 0)
        self._write_record(0, 0, payload)
        sid = self._next_id; self._next_id += 1
        self._write_record(sid, 0, schema_str.encode("utf-8"))
        return sid
    def start_entry(self, name, type_str):
        if not name.startswith("/"): name = "/" + name
        eid = self._next_id; self._next_id += 1
        payload = struct.pack("<BI", 0, eid) + struct.pack("<I", len(name)) + name.encode("utf-8") + struct.pack("<I", len(type_str)) + type_str.encode("utf-8") + struct.pack("<I", 0)
        self._write_record(0, 0, payload)
        return eid
    def write_pose2d(self, eid, ts, x_m, y_m, rot):
        self._write_record(eid, ts, struct.pack("<ddd", x_m, y_m, rot))
    def write_boolean(self, eid, ts, val):
        self._write_record(eid, ts, struct.pack("<B", 1 if val else 0))
    def close(self):
        self._fh.close()

# ─────────────────────────────────────────────
# Tracking Logic
# ─────────────────────────────────────────────
@dataclass
class RobotPose:
    x_in: float = 0.0; y_in: float = 0.0; heading: float = 0.0; visible: bool = False
    @property
    def x_m(self): return self.x_in * 0.0254
    @property
    def y_m(self): return self.y_in * 0.0254

class RobotTracker:
    """
    MOG2 background subtraction + contour blobs + greedy nearest-neighbour
    ID assignment so robot labels stay consistent across frames.
    """
 
    NUM_ROBOTS = 4
 
    def __init__(self, cv2, np):
        self.cv2 = cv2
        self.np  = np
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=32, detectShadows=False
        )
        self.tracked_poses = [RobotPose() for _ in range(4)]
        self.prev_centers  = [None] * 4
        self._initialized  = False
        self._warmup_frames = 30
 
    def warmup(self, frame):
        self.bg_subtractor.apply(frame)
 
    def update(self, frame, H):
        cv2, np = self.cv2, self.np
 
        fg_mask = self.bg_subtractor.apply(frame)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
 
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
        h, w       = frame.shape[:2]
        frame_area = h * w
 
        robot_blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if frame_area * 0.003 < area < frame_area * 0.10:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    robot_blobs.append((cx, cy, area, cnt))
 
        robot_blobs.sort(key=lambda b: b[2], reverse=True)
        robot_blobs = robot_blobs[:self.NUM_ROBOTS]
 
        if not robot_blobs:
            for p in self.tracked_poses:
                p.visible = False
            return self.tracked_poses
 
        detections_field = []
        for cx, cy, area, cnt in robot_blobs:
            if H is not None:
                denom = H[2, 0]*cx + H[2, 1]*cy + H[2, 2]
                if denom != 0:
                    fx = (H[0, 0]*cx + H[0, 1]*cy + H[0, 2]) / denom
                    fy = (H[1, 0]*cx + H[1, 1]*cy + H[1, 2]) / denom
                else:
                    fx, fy = cx, cy
            else:
                fx, fy = cx, cy
 
            heading = math.radians(cv2.fitEllipse(cnt)[2]) if len(cnt) >= 5 else 0.0
            detections_field.append((fx, fy, heading))
 
        if not self._initialized or all(p is None for p in self.prev_centers):
            detections_field.sort(key=lambda d: (d[1], d[0]))
            for i, det in enumerate(detections_field):
                self.tracked_poses[i].x_in    = det[0]
                self.tracked_poses[i].y_in    = det[1]
                self.tracked_poses[i].heading = det[2]
                self.tracked_poses[i].visible = True
                self.prev_centers[i] = (det[0], det[1])
            for i in range(len(detections_field), self.NUM_ROBOTS):
                self.tracked_poses[i].visible = False
            self._initialized = True
        else:
            used = [False] * len(detections_field)
            for i in range(self.NUM_ROBOTS):
                prev = self.prev_centers[i]
                if prev is None:
                    continue
                best_dist = float("inf")
                best_j    = -1
                for j, det in enumerate(detections_field):
                    if used[j]:
                        continue
                    d = math.hypot(det[0] - prev[0], det[1] - prev[1])
                    if d < best_dist and d < 25.0:
                        best_dist = d
                        best_j    = j
                if best_j >= 0:
                    used[best_j] = True
                    det = detections_field[best_j]
                    self.tracked_poses[i].x_in    = det[0]
                    self.tracked_poses[i].y_in    = det[1]
                    self.tracked_poses[i].heading = det[2]
                    self.tracked_poses[i].visible = True
                    self.prev_centers[i] = (det[0], det[1])
                else:
                    self.tracked_poses[i].visible = False
 
        return self.tracked_poses

# ─────────────────────────────────────────────
# Field Detector
# ─────────────────────────────────────────────
class FieldDetector:
    def __init__(self, cv2, np):
        self.cv2 = cv2
        self.np = np

    def detect_field(self, frame):
        cv2, np = self.cv2, self.np
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:10]:
            if cv2.contourArea(cnt) < (h * w * 0.05): continue
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1).flatten()
                ordered = np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)
                dst = np.array([[0, 144.0], [144.0, 144.0], [144.0, 0], [0, 0]], dtype=np.float32)
                H, _ = cv2.findHomography(ordered, dst)
                return ordered, H
        return None

# ─────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────
def process_match(
    video_path,
    output_dir,
    start_offset_sec=0.0,
    sample_rate_fps=10.0,
    debug=False,
    manual_corners_px=None,
):
    cv2 = _require("cv2", "opencv-python")
    np  = _require("numpy")
 
    os.makedirs(output_dir, exist_ok=True)
 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video: {}".format(video_path))
        sys.exit(1)
 
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps
    print("[INFO] Video: {:.1f} fps, {} frames, {:.1f}s".format(
        video_fps, total_frames, duration_sec))
 
    start_frame = int(start_offset_sec * video_fps)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
 
    frame_step = max(1, int(round(video_fps / sample_rate_fps)))
    print("[INFO] Processing every {} frames (~{:.1f} fps output)".format(
        frame_step, video_fps / frame_step))
 
    # ── output paths ─────────────────────────────────────────────────
    csv_path    = os.path.join(output_dir, "robot_positions.csv")
    wpilog_path = os.path.join(output_dir, "match_log.wpilog")
    debug_dir   = os.path.join(output_dir, "tracker_debug") if debug else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
 
    # ── WPILOG setup ─────────────────────────────────────────────────
    log = WPILogWriter(wpilog_path)
 
    # Register schemas FIRST so AdvantageScope can decode the structs
    log._register_schema("Rotation2d", WPILogWriter.ROTATION2D_SCHEMA)
    log._register_schema("Pose2d",     WPILogWriter.POSE2D_SCHEMA)
 
    # Register one Pose2d entry and one boolean visibility entry per robot
    robot_entries = []
    vis_entries   = []
    for i in range(4):
        robot_entries.append(log.start_entry(
            name     = "Robot{}/Pose".format(i),
            type_str = "struct:Pose2d",
            # metadata = "",
        ))
        vis_entries.append(log.start_entry(
            name     = "Robot{}/Visible".format(i),
            type_str = "boolean",
            # metadata = "",
        ))
 
    # ── CSV setup ────────────────────────────────────────────────────
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp_s",
        "robot0_x_in", "robot0_y_in", "robot0_heading_rad", "robot0_visible",
        "robot1_x_in", "robot1_y_in", "robot1_heading_rad", "robot1_visible",
        "robot2_x_in", "robot2_y_in", "robot2_heading_rad", "robot2_visible",
        "robot3_x_in", "robot3_y_in", "robot3_heading_rad", "robot3_visible",
    ])
 
    # ── detectors ────────────────────────────────────────────────────
    field_detector = FieldDetector(cv2, np)
    robot_tracker  = RobotTracker(cv2, np)
 
    H             = None
    field_corners = None
 
    if manual_corners_px is not None:
        bl, br, tr, tl = [np.array(c, dtype=np.float32) for c in manual_corners_px]
        ordered    = np.array([tl, tr, br, bl], dtype=np.float32)
        field_size = 144.0
        dst = np.array([
            [0,          field_size],
            [field_size, field_size],
            [field_size, 0],
            [0,          0],
        ], dtype=np.float32)
        H, _ = cv2.findHomography(ordered, dst)
        field_corners = ordered
        print("[INFO] Using manually specified field corners.")
 
    warmup_count   = 0
    frame_num      = start_frame   # index of the NEXT frame to be read
    processed      = 0             # number of frames actually tracked
    last_timestamp = 0             # track highest timestamp for finish records
 
    print("[INFO] Starting processing... (warming up background model)")
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        # Timestamp of this frame in the original video
        current_sec   = frame_num / video_fps
        # Timestamp relative to match start (what we log)
        match_time_s  = current_sec - start_offset_sec
        timestamp_us  = max(0, int(match_time_s * 1000000))
 
        frame_num += 1  # advance AFTER computing current_sec
 
        # ── warmup phase: feed every frame to the BG model ──────────
        if warmup_count < robot_tracker._warmup_frames:
            robot_tracker.warmup(frame)
            warmup_count += 1
            continue
 
        # ── only process every Nth frame ────────────────────────────
        elapsed = frame_num - start_frame - robot_tracker._warmup_frames
        if elapsed % frame_step != 0:
            continue
 
        # ── auto-detect field corners ────────────────────────────────
        if H is None or (manual_corners_px is None and processed % 300 == 0):
            result = field_detector.detect_field(frame)
            if result is not None:
                field_corners, H = result
                if processed == 0:
                    print("  [t={:.1f}s] Field detected.".format(match_time_s))
 
        # ── track robots ─────────────────────────────────────────────
        poses = robot_tracker.update(frame, H)
        last_timestamp = timestamp_us
 
        # ── write WPILOG ─────────────────────────────────────────────
        for i, pose in enumerate(poses):
            log.write_pose2d(robot_entries[i], timestamp_us,
                             pose.x_m, pose.y_m, pose.heading)
            log.write_boolean(vis_entries[i], timestamp_us, pose.visible)
 
        # ── write CSV ────────────────────────────────────────────────
        row = ["{:.4f}".format(match_time_s)]
        for pose in poses:
            row += [
                "{:.2f}".format(pose.x_in),
                "{:.2f}".format(pose.y_in),
                "{:.4f}".format(pose.heading),
                "1" if pose.visible else "0",
            ]
        csv_writer.writerow(row)
 
        # ── debug frames ─────────────────────────────────────────────
        if debug and debug_dir:
            debug_frame = frame.copy()
            if field_corners is not None:
                cv2.polylines(debug_frame, [field_corners.astype(int)],
                              True, (255, 255, 0), 2)
            for i, pose in enumerate(poses):
                if pose.visible and H is not None:
                    H_inv = np.linalg.inv(H)
                    denom = (H_inv[2,0]*pose.x_in +
                             H_inv[2,1]*pose.y_in + H_inv[2,2])
                    if denom != 0:
                        px_d = int((H_inv[0,0]*pose.x_in +
                                    H_inv[0,1]*pose.y_in + H_inv[0,2]) / denom)
                        py_d = int((H_inv[1,0]*pose.x_in +
                                    H_inv[1,1]*pose.y_in + H_inv[1,2]) / denom)
                        cv2.circle(debug_frame, (px_d, py_d), 20, colors[i], 3)
                        cv2.putText(debug_frame, "R{}".format(i),
                                    (px_d - 10, py_d - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)
                        dx = int(30 * math.cos(pose.heading))
                        dy = int(-30 * math.sin(pose.heading))
                        cv2.arrowedLine(debug_frame, (px_d, py_d),
                                        (px_d + dx, py_d + dy), colors[i], 2)
            cv2.putText(debug_frame, "t={:.2f}s".format(match_time_s),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imwrite(os.path.join(debug_dir,
                        "frame_{:06d}.jpg".format(processed)), debug_frame)
 
        processed += 1
        if processed % 50 == 0:
            span = float(max(1, total_frames - start_frame))
            pct  = (frame_num - start_frame) / span * 100
            print("  [{:.0f}%] t={:.1f}s | Robots visible: {}/4".format(
                pct, match_time_s,
                sum(1 for p in poses if p.visible)))
 
    # ── finalise ─────────────────────────────────────────────────────
    # Use last_timestamp so the timeline spans the full match duration
    end_ts = last_timestamp
    # for i in range(4):
    #     log.finish_entry(robot_entries[i], end_ts)
    #     log.finish_entry(vis_entries[i],   end_ts)
 
    log.close()
    csv_file.close()
    cap.release()
 
    print("\n[DONE] Processed {} frames.".format(processed))
    print("  CSV log: {}".format(csv_path))
    print("  WPILOG:  {}".format(wpilog_path))
    if debug:
        print("  Debug:   {}".format(debug_dir))
    print_advantagescope_instructions()
 
 
def print_advantagescope_instructions():
    print("""
+------------------------------------------------------------------+
|         AdvantageScope Visualization Instructions                |
+------------------------------------------------------------------+
|  1. Open AdvantageScope                                          |
|  2. File > Open Log(s)... > select match_log.wpilog              |
|  3. Click the "+" tab button and choose "2D Field"               |
|  4. Set Field to "FTC DECODE" (or nearest FTC season)            |
|  5. Drag these keys into the Poses section:                      |
|       Robot0/Pose  -> Blue 1                                     |
|       Robot1/Pose  -> Blue 2                                     |
|       Robot2/Pose  -> Red 1                                      |
|       Robot3/Pose  -> Red 2                                      |
|  6. Right-click each to change color / shape                     |
|  7. Press play to watch the match replay!                        |
|                                                                  |
|  TIP: Use --debug to verify robot ID assignments visually.       |
|  If robots are misidentified, use --start-offset to begin after  |
|  tip-off, or use ftc_calibrate.py with --corners.                |
+------------------------------------------------------------------+
""")
# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Track FTC DECODE robots from a YouTube match video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("url", nargs="?",
                        help="YouTube URL of the match")
    parser.add_argument("--output-dir",   default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--start-offset", type=float, default=0.0,
                        help="Seconds to skip at start before match begins")
    parser.add_argument("--sample-rate",  type=float, default=10.0,
                        help="Frames per second to process (default: 10)")
    parser.add_argument("--debug",        action="store_true",
                        help="Save annotated debug frames to tracker_debug/")
    parser.add_argument("--no-download",  action="store_true",
                        help="Skip YouTube download (requires --video-path)")
    parser.add_argument("--video-path",   default=None,
                        help="Path to a local video file")
    parser.add_argument("--corners",      default=None,
                        help="Path to field_corners.json from ftc_calibrate.py")
 
    args = parser.parse_args()
 
    if args.no_download:
        if not args.video_path:
            parser.error("--no-download requires --video-path")
        video_path = args.video_path
    else:
        if not args.url:
            parser.error("A YouTube URL is required (or use --no-download --video-path)")
        os.makedirs(args.output_dir, exist_ok=True)
        video_path = download_video(args.url, args.output_dir)
 
    print("[INFO] Output directory: {}".format(args.output_dir))
 
    manual_corners = None
    if args.corners:
        with open(args.corners) as f:
            cdata = json.load(f)
        manual_corners = cdata["corners_px"]
        print("[INFO] Loaded manual corners from: {}".format(args.corners))
 
    process_match(
        video_path        = video_path,
        output_dir        = args.output_dir,
        start_offset_sec  = args.start_offset,
        sample_rate_fps   = args.sample_rate,
        debug             = args.debug,
        manual_corners_px = manual_corners,
    )
 
 
if __name__ == "__main__":
    main()