# -*- coding: utf-8 -*-
"""
auto_scout.py — FTC DECODE robot tracker.

Pipeline:
  1. Build a per-pixel MEDIAN background image from 80 frames sampled evenly
     across the full match.  Because robots move around, each floor pixel has
     the floor colour in >50% of frames → the median is a clean empty-field
     image with no ghost robots.

  2. Each frame: diff against the median background, threshold, find contours
     inside the field polygon, take the 4 LARGEST blobs.  Robots are always
     the biggest non-floor objects.  No BEV needed — works in the original
     camera image, so perspective distortion doesn't corrupt dark robots.

  3. Convert blob centroids to field coordinates via the homography from the
     calibrated field corners.

  4. Greedy nearest-neighbour assignment to 4 persistent tracks (no velocity
     prediction — avoids drift when blobs temporarily merge).  Tracks coast
     for up to MAX_COAST frames when no blob is assigned.

Usage:
  python3 auto_scout.py --no-download --video-path match.mp4 \\
      --corners field_corners.json [--debug] [--start-offset 1.0]

field_corners.json is produced by ftc_calibrate.py.
"""

import argparse
import csv
import json
import math
import os
import struct
import sys
from dataclasses import dataclass


def _require(package, pip_name=None):
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        name = pip_name or package
        print("[ERROR] Missing: {}\n  Install: pip install {}".format(name, name))
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# WPILOG writer  (double[] — no struct schema needed)
# ─────────────────────────────────────────────────────────────────────────────
class WPILogWriter:
    HEADER_MAGIC = b"WPILOG"

    def __init__(self, path):
        self._fh = open(path, "wb")
        self._next_id = 1
        self._fh.write(self.HEADER_MAGIC)
        self._fh.write(struct.pack("<BBI", 1, 0, 0))

    def _encode_int(self, v, max_bytes):
        for n in [1, 2, 4, 8]:
            if n > max_bytes:
                break
            if v < (1 << (8 * n)):
                return v.to_bytes(n, "little"), n
        return v.to_bytes(max_bytes, "little"), max_bytes

    def _write_record(self, eid, ts, data):
        eb, el = self._encode_int(eid,       4)
        sb, sl = self._encode_int(len(data), 4)
        tb, tl = self._encode_int(ts,        8)
        bf = ((el-1)&3) | (((sl-1)&3)<<2) | (((tl-1)&7)<<4)
        self._fh.write(struct.pack("<B", bf))
        self._fh.write(eb); self._fh.write(sb); self._fh.write(tb)
        self._fh.write(data)

    def start_entry(self, name, type_str):
        if not name.startswith("/"):
            name = "/" + name
        eid = self._next_id; self._next_id += 1
        nb = name.encode(); tb = type_str.encode(); mb = b""
        payload = (struct.pack("<BI", 0, eid) +
                   struct.pack("<I", len(nb)) + nb +
                   struct.pack("<I", len(tb)) + tb +
                   struct.pack("<I", len(mb)) + mb)
        self._write_record(0, 0, payload)
        return eid

    def write_pose2d(self, eid, ts, x_m, y_m, rot_rad):
        self._write_record(eid, ts,
            struct.pack("<ddd", float(x_m), float(y_m), float(rot_rad)))

    def write_boolean(self, eid, ts, val):
        self._write_record(eid, ts, struct.pack("<B", 1 if val else 0))

    def close(self):
        self._fh.close()


# ─────────────────────────────────────────────────────────────────────────────
# RobotPose
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RobotPose:
    x_in:    float = 0.0
    y_in:    float = 0.0
    heading: float = 0.0
    visible: bool  = False

    @property
    def x_m(self): return self.x_in * 0.0254

    @property
    def y_m(self): return self.y_in * 0.0254


# ─────────────────────────────────────────────────────────────────────────────
# RobotTracker
# ─────────────────────────────────────────────────────────────────────────────
class RobotTracker:
    """
    Detect robots by diffing each frame against a per-pixel median background,
    then assign the 4 largest foreground blobs to 4 persistent tracks using
    greedy nearest-neighbour.

    Key design decisions:
    - Per-pixel median of 80 evenly-sampled frames → clean empty-field BG.
      Robots move enough that the floor shows through at every pixel.
    - Work in ORIGINAL image space (not BEV) so perspective distortion doesn't
      make dark robots invisible.
    - NO velocity prediction in the tracker — pure last-known-position memory.
      Avoids compounding drift when two robots temporarily merge into one blob.
    - Tracks coast for MAX_COAST frames so robots that stop briefly stay visible.
    """

    N_BG_SAMPLES  = 80    # frames sampled to build the median background
    FG_THRESH     = 25    # grayscale diff threshold (0-255)
    BLOB_MIN      = 300   # minimum blob area in image pixels²
    KERNEL_PX     = 9     # morphology kernel size
    MAX_COAST     = 60    # frames a track can coast without a detection
    MAX_DIST_IN   = 60.0  # max field-inch jump for blob→track assignment

    def __init__(self, cv2, np):
        self.cv2, self.np = cv2, np
        self._bg          = None   # uint8 BGR background image (original space)
        self._field_mask  = None   # uint8 mask in original image space
        self._H_2d        = None   # image px → field inches
        self._kern        = cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE, (self.KERNEL_PX, self.KERNEL_PX))

        self.tracked_poses = [RobotPose() for _ in range(4)]
        self._pos    = [None] * 4   # last known (fx_in, fy_in) per robot
        self._coast  = [999]  * 4   # 999 = never seen
        self._initialized = False

    # ── public API ──────────────────────────────────────────────────────────

    def setup(self, video_path, ordered_corners, frame_shape):
        """
        Build the field mask, homography, and median background.

        ordered_corners : float32 [TL, TR, BR, BL] in image pixels
        frame_shape     : (h, w, c) of a typical frame
        """
        cv2, np = self.cv2, self.np
        h, w = frame_shape[:2]

        # Field-polygon mask in image space — slightly inset from the corners
        # to avoid alliance-station wall bleed on the left/right edges.
        tl, tr, br, bl = ordered_corners
        poly = np.array([
            [int(tl[0]+15), int(tl[1]+5)],
            [int(tr[0]-15), int(tr[1]+5)],
            [int(br[0]-15), int(br[1]-5)],
            [int(bl[0]+15), int(bl[1]-5)],
        ], dtype=np.int32)
        self._field_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self._field_mask, [poly], 255)

        # Homography: image pixels → field inches
        dst2d = np.array([[0,144],[144,144],[144,0],[0,0]], dtype=np.float32)
        self._H_2d, _ = cv2.findHomography(ordered_corners, dst2d)

        # Build per-pixel median background
        print("[INFO] Building median background ({} frames)…".format(
            self.N_BG_SAMPLES))
        cap2 = cv2.VideoCapture(video_path)
        n_total = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        indices  = np.linspace(0, n_total - 1, self.N_BG_SAMPLES, dtype=int)
        frames   = []
        for idx in indices:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, f = cap2.read()
            if ret:
                frames.append(f.astype(np.float32))
        cap2.release()

        if frames:
            self._bg = np.median(frames, axis=0).astype(np.uint8)
        else:
            self._bg = np.zeros((h, w, 3), dtype=np.uint8)
        print("[INFO] Background ready.")

    def update(self, frame, _H=None):
        """Process one frame and return 4 RobotPoses."""
        if self._bg is None or self._H_2d is None:
            for p in self.tracked_poses:
                p.visible = False
            return self.tracked_poses

        blobs = self._get_blobs(frame)

        # ── initialise on first frame with ≥ 4 blobs ────────────────────
        if not self._initialized:
            if len(blobs) < 4:
                for p in self.tracked_poses:
                    p.visible = False
                return self.tracked_poses
            # Assign left → right by field X for stable IDs
            for i, b in enumerate(sorted(blobs[:4], key=lambda b: b[0])):
                self._pos[i]   = (b[0], b[1])
                self._coast[i] = 0
                self.tracked_poses[i].x_in    = b[0]
                self.tracked_poses[i].y_in    = b[1]
                self.tracked_poses[i].heading = b[2]
                self.tracked_poses[i].visible = True
            self._initialized = True
            return self.tracked_poses

        # ── greedy nearest-neighbour assignment ──────────────────────────
        used = set()
        for i in range(4):
            best_d, best_j = self.MAX_DIST_IN, -1
            if self._pos[i] is None:
                for j in range(len(blobs)):
                    if j not in used:
                        best_j = j
                        break
            else:
                px, py = self._pos[i]
                for j, (bx, by, _) in enumerate(blobs):
                    if j in used:
                        continue
                    d = math.hypot(bx - px, by - py)
                    if d < best_d:
                        best_d, best_j = d, j

            if best_j >= 0:
                bx, by, heading = blobs[best_j]
                self._pos[i]   = (bx, by)
                self._coast[i] = 0
                self.tracked_poses[i].x_in    = bx
                self.tracked_poses[i].y_in    = by
                self.tracked_poses[i].heading = heading
                self.tracked_poses[i].visible = True
                used.add(best_j)
            else:
                self._coast[i] += 1
                self.tracked_poses[i].visible = (
                    self._coast[i] <= self.MAX_COAST
                    and self._pos[i] is not None)
                if self._pos[i] is not None:
                    self.tracked_poses[i].x_in = self._pos[i][0]
                    self.tracked_poses[i].y_in = self._pos[i][1]

        return self.tracked_poses

    # ── blob detector ────────────────────────────────────────────────────

    def _foreground_contours(self, frame):
        """
        Return (all_contours, top4_contours) from the foreground mask.
        Used by the debug overlay to draw every detected blob.
        all_contours: every contour above BLOB_MIN, sorted area desc.
        top4_contours: the same contours that _get_blobs() would pick.
        """
        cv2, np = self.cv2, self.np
        diff = cv2.absdiff(frame, self._bg)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, fg = cv2.threshold(gray, self.FG_THRESH, 255, cv2.THRESH_BINARY)
        fg = cv2.bitwise_and(fg, self._field_mask)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kern)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self._kern)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = sorted(
            [c for c in cnts if cv2.contourArea(c) >= self.BLOB_MIN],
            key=lambda c: -cv2.contourArea(c))
        return valid, valid[:4]

    def _get_blobs(self, frame):
        """
        Return up to 4 (fx_in, fy_in, heading_rad) tuples, sorted largest first.
        Works in original image space — no BEV needed.
        """
        cv2, np = self.cv2, self.np

        diff = cv2.absdiff(frame, self._bg)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, fg = cv2.threshold(gray, self.FG_THRESH, 255, cv2.THRESH_BINARY)
        fg = cv2.bitwise_and(fg, self._field_mask)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kern)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self._kern)

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < self.BLOB_MIN:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            # Project image centroid → field inches
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            fp = cv2.perspectiveTransform(pt, self._H_2d)[0][0]
            fx, fy = float(fp[0]), float(fp[1])
            # Discard points outside the 144×144 in field bounds
            if not (0 <= fx <= 144 and 0 <= fy <= 144):
                continue
            heading = math.radians(cv2.fitEllipse(c)[2]) if len(c) >= 5 else 0.0
            blobs.append((fx, fy, heading, area))

        blobs.sort(key=lambda b: -b[3])
        return [(fx, fy, h) for fx, fy, h, _ in blobs[:4]]


# ─────────────────────────────────────────────────────────────────────────────
# Field corner auto-detector (fallback when --corners not provided)
# ─────────────────────────────────────────────────────────────────────────────
class FieldDetector:
    def __init__(self, cv2, np):
        self.cv2, self.np = cv2, np

    def detect_field(self, frame):
        cv2, np = self.cv2, self.np
        h, w = frame.shape[:2]
        edges = cv2.Canny(
            cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5,5), 0),
            30, 100)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
            if cv2.contourArea(cnt) < h * w * 0.05:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                pts  = approx.reshape(4, 2).astype(np.float32)
                s    = pts.sum(axis=1)
                diff = np.diff(pts, axis=1).flatten()
                ordered = np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                                    pts[np.argmax(s)], pts[np.argmax(diff)]],
                                   dtype=np.float32)
                H, _ = cv2.findHomography(
                    ordered,
                    np.array([[0,144],[144,144],[144,0],[0,0]], np.float32))
                return ordered, H
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ─────────────────────────────────────────────────────────────────────────────
def process_match(
    video_path,
    output_dir,
    start_offset_sec  = 0.0,
    sample_rate_fps   = 10.0,
    debug             = False,
    manual_corners_px = None,
):
    cv2 = _require("cv2", "opencv-python")
    np  = _require("numpy")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open: {}".format(video_path)); sys.exit(1)

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("[INFO] Video: {:.1f} fps, {} frames, {:.1f}s".format(
        video_fps, total_frames, total_frames / video_fps))

    start_frame = int(start_offset_sec * video_fps)
    frame_step  = max(1, int(round(video_fps / sample_rate_fps)))
    print("[INFO] Processing every {} frames (~{:.1f} fps output)".format(
        frame_step, video_fps / frame_step))

    # ── output files ─────────────────────────────────────────────────────
    csv_path    = os.path.join(output_dir, "robot_positions.csv")
    wpilog_path = os.path.join(output_dir, "match_log.wpilog")
    debug_dir   = os.path.join(output_dir, "tracker_debug") if debug else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    log = WPILogWriter(wpilog_path)
    pose_eids = [log.start_entry("Robot{}/Pose".format(i), "double[]") for i in range(4)]
    vis_eids  = [log.start_entry("Robot{}/Visible".format(i), "boolean") for i in range(4)]

    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp_s",
        "robot0_x_in","robot0_y_in","robot0_heading_rad","robot0_visible",
        "robot1_x_in","robot1_y_in","robot1_heading_rad","robot1_visible",
        "robot2_x_in","robot2_y_in","robot2_heading_rad","robot2_visible",
        "robot3_x_in","robot3_y_in","robot3_heading_rad","robot3_visible",
    ])

    # ── set up tracker ───────────────────────────────────────────────────
    tracker        = RobotTracker(cv2, np)
    field_detector = FieldDetector(cv2, np)

    ordered = None
    H_2d    = None
    H_inv   = None

    # Read one frame to get shape
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, sample_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame."); sys.exit(1)
    frame_shape = sample_frame.shape

    if manual_corners_px is not None:
        bl, br, tr, tl = [np.array(c, np.float32) for c in manual_corners_px]
        ordered = np.array([tl, tr, br, bl], np.float32)
        H_2d, _ = cv2.findHomography(
            ordered, np.array([[0,144],[144,144],[144,0],[0,0]], np.float32))
        H_inv = np.linalg.inv(H_2d)
        tracker.setup(video_path, ordered, frame_shape)
        print("[INFO] Manual corners loaded.")
        # Save median background for debugging
        if tracker._bg is not None:
            bg_path = os.path.join(output_dir, "median_background.jpg")
            cv2.imwrite(bg_path, tracker._bg)
            print("[INFO] Median background saved: {}".format(bg_path))
    else:
        print("[WARN] No corners — will attempt auto-detection.")

    # ── main loop ────────────────────────────────────────────────────────
    frame_num = start_frame
    processed = 0
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    debug_colors = [(220,160,0),(0,220,255),(0,0,220),(0,120,255)]
    print("[INFO] Processing…")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        match_time_s = frame_num / video_fps - start_offset_sec
        timestamp_us = max(0, int(match_time_s * 1_000_000))
        frame_num   += 1

        if (frame_num - start_frame) % frame_step != 0:
            continue

        # Auto-detect corners if not provided
        if ordered is None:
            result = field_detector.detect_field(frame)
            if result is not None:
                ordered, H_2d = result
                H_inv = np.linalg.inv(H_2d)
                tracker.setup(video_path, ordered, frame.shape)
                print("  [t={:.1f}s] Field auto-detected.".format(match_time_s))
                if tracker._bg is not None:
                    bg_path = os.path.join(output_dir, "median_background.jpg")
                    cv2.imwrite(bg_path, tracker._bg)

        poses = tracker.update(frame)

        # ── write outputs ─────────────────────────────────────────────
        for i, p in enumerate(poses):
            log.write_pose2d(pose_eids[i], timestamp_us, p.x_m, p.y_m, p.heading)
            log.write_boolean(vis_eids[i], timestamp_us, p.visible)

        row = ["{:.4f}".format(match_time_s)]
        for p in poses:
            row += ["{:.2f}".format(p.x_in), "{:.2f}".format(p.y_in),
                    "{:.4f}".format(p.heading), "1" if p.visible else "0"]
        csv_writer.writerow(row)

        # ── debug frame ───────────────────────────────────────────────
        if debug and debug_dir and H_inv is not None:
            dbg = frame.copy()
            if ordered is not None:
                cv2.polylines(dbg,
                              [ordered.astype(np.int32).reshape(-1,1,2)],
                              True, (0, 200, 0), 2)

            # Draw all foreground blobs (white outline = ignored, yellow = top-4)
            if tracker._bg is not None:
                all_cnts, top4_cnts = tracker._foreground_contours(frame)
                top4_set = set(id(c) for c in top4_cnts)
                for c in all_cnts:
                    area = cv2.contourArea(c)
                    is_top4 = id(c) in top4_set
                    color = (0, 255, 255) if is_top4 else (180, 180, 180)
                    thickness = 2 if is_top4 else 1
                    cv2.drawContours(dbg, [c], -1, color, thickness)
                    # Label area on larger blobs
                    if area >= tracker.BLOB_MIN * 2:
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            bx = int(M["m10"] / M["m00"])
                            by = int(M["m01"] / M["m00"])
                            cv2.putText(dbg, "{:.0f}".format(area),
                                        (bx - 15, by + 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                        color, 1)

            n_drawn = 0
            fh, fw = frame.shape[:2]
            for i, p in enumerate(poses):
                if not p.visible:
                    continue
                pt  = np.array([[[p.x_in, p.y_in]]], dtype=np.float32)
                img = cv2.perspectiveTransform(pt, H_inv)[0][0]
                ix, iy = int(img[0]), int(img[1])
                if not (0 <= ix < fw and 0 <= iy < fh):
                    continue
                col = debug_colors[i]
                cv2.circle(dbg, (ix, iy), 16, col, -1)
                cv2.circle(dbg, (ix, iy), 19, (255, 255, 255), 2)
                cv2.putText(dbg, "R{}".format(i), (ix - 10, iy - 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                dx = int(28 * math.cos(p.heading))
                dy = int(-28 * math.sin(p.heading))
                cv2.arrowedLine(dbg, (ix, iy), (ix+dx, iy+dy), (255,255,255), 2)
                n_drawn += 1

            lbl = "init" if not tracker._initialized else "{}/4".format(n_drawn)
            cv2.putText(dbg, "t={:.2f}s  [{}]".format(match_time_s, lbl),
                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            if processed % 5 == 0:
                cv2.imwrite(os.path.join(debug_dir,
                            "frame_{:06d}.jpg".format(processed)), dbg)

        processed += 1
        if processed % 50 == 0:
            pct = (frame_num - start_frame) / max(1, total_frames - start_frame) * 100
            print("  [{:.0f}%] t={:.1f}s | visible: {}/4".format(
                pct, match_time_s, sum(1 for p in poses if p.visible)))

    log.close(); csv_file.close(); cap.release()
    print("\n[DONE] {} frames processed.".format(processed))
    print("  CSV:    {}".format(csv_path))
    print("  WPILOG: {}".format(wpilog_path))
    if debug:
        print("  Debug:  {}".format(debug_dir))
    _print_instructions()


def _print_instructions():
    print("""
+------------------------------------------------------------------+
|         AdvantageScope Visualization Instructions                |
+------------------------------------------------------------------+
|  1. Open AdvantageScope                                          |
|  2. File > Open Log(s) … > select match_log.wpilog               |
|  3. "+" tab → 2D Field → set Field to FTC DECODE season          |
|  4. Drag Robot0/Pose into Poses                                  |
|  5. Click icon LEFT of the field name →                          |
|       Format: Pose2d   Units: Meters + Radians                   |
|  6. Repeat for Robot1, Robot2, Robot3                            |
|  7. Press play!                                                  |
|                                                                  |
|  Robot IDs: assigned left-to-right at match start.               |
|  Use --debug to generate annotated frames for verification.       |
+------------------------------------------------------------------+
""")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def download_video(url, output_dir):
    import subprocess
    out = os.path.join(output_dir, "match_video.mp4")
    print("[INFO] Downloading:", url)
    subprocess.run(["yt-dlp", "-f", "best[ext=mp4]", "-o", out, url], check=True)
    return out


def main():
    p = argparse.ArgumentParser(
        description="Track FTC DECODE robots from a match video.")
    p.add_argument("url",            nargs="?", help="YouTube URL")
    p.add_argument("--output-dir",   default="./output")
    p.add_argument("--start-offset", type=float, default=0.0,
                   help="Seconds to skip before match timer starts")
    p.add_argument("--sample-rate",  type=float, default=10.0,
                   help="Frames per second to process (default 10)")
    p.add_argument("--debug",        action="store_true",
                   help="Save annotated debug frames to tracker_debug/")
    p.add_argument("--no-download",  action="store_true")
    p.add_argument("--video-path",   default=None)
    p.add_argument("--corners",      default=None,
                   help="field_corners.json from ftc_calibrate.py")
    args = p.parse_args()

    if args.no_download:
        if not args.video_path:
            p.error("--no-download requires --video-path")
        video_path = args.video_path
    else:
        if not args.url:
            p.error("YouTube URL required (or --no-download --video-path)")
        os.makedirs(args.output_dir, exist_ok=True)
        video_path = download_video(args.url, args.output_dir)

    print("[INFO] Output:", args.output_dir)
    corners = None
    if args.corners:
        with open(args.corners) as f:
            corners = json.load(f)["corners_px"]
        print("[INFO] Corners:", args.corners)

    process_match(
        video_path        = video_path,
        output_dir        = args.output_dir,
        start_offset_sec  = args.start_offset,
        sample_rate_fps   = args.sample_rate,
        debug             = args.debug,
        manual_corners_px = corners,
    )


if __name__ == "__main__":
    main()