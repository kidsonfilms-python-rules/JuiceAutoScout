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
    MIN_RADIUS_PX = 10    # minimum inscribed-circle radius to be a robot
                          # balls ≤8 px, robots ≥11 px — clean gap
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

        # Field-polygon mask in image space.
        # Robots at the field edges hang slightly outside the calibrated
        # corner polygon (especially at the bottom where the camera angle
        # is steep). We expand the polygon outward per-edge so the full
        # robot blob is captured, while keeping the top edge tight to
        # avoid picking up the alliance-station backdrops above the field.
        tl, tr, br, bl = ordered_corners
        cx_poly = (tl[0]+tr[0]+br[0]+bl[0]) / 4
        cy_poly = (tl[1]+tr[1]+br[1]+bl[1]) / 4

        SIDE_PAD   = 25   # px — left / right expansion
        BOTTOM_PAD = 25   # px — bottom expansion (robots overhang here most)
        TOP_PAD    =  5   # px — top kept tight to avoid alliance backdrops

        def _pad(x, y):
            dx = x - cx_poly; dy = y - cy_poly
            px = SIDE_PAD   if dx > 0 else -SIDE_PAD
            py = BOTTOM_PAD if dy > 0 else -TOP_PAD
            return int(x + px), int(y + py)

        poly = np.array([_pad(*tl), _pad(*tr), _pad(*br), _pad(*bl)],
                        dtype=np.int32)
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
        Return (all_contours, split_centers) from the foreground mask.
        all_contours : every contour above BLOB_MIN, sorted area desc.
                       Used to draw white/yellow outlines.
        split_centers: list of (cx_px, cy_px) — the actual sub-centers
                       produced by _split_contour(), i.e. what _get_blobs
                       would use.  Drawn as small crosses in the overlay.
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
        # Collect split centers for all valid contours
        all_split = []
        for c in valid:
            area = float(cv2.contourArea(c))
            subs = self._split_contour(c)
            per = area / max(len(subs), 1)
            for (sx, sy) in subs:
                all_split.append((sx, sy, per))
        all_split.sort(key=lambda x: -x[2])
        split_centers = [(int(sx), int(sy)) for sx, sy, _ in all_split]
        return valid, split_centers

    def _split_contour(self, contour):
        """
        Use the distance transform to find individual robot centers within a
        contour that may contain multiple touching/merged robots.

        Returns a list of (cx_px, cy_px) image-pixel centers — one per
        local maximum found in the distance transform of the filled contour.
        A single isolated robot produces exactly one peak; two touching robots
        produce two peaks; etc.
        """
        cv2, np = self.cv2, self.np
        h, w = self._field_mask.shape[:2]

        # Rasterise just this contour into its own mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Distance transform: each pixel's value = distance to nearest edge
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist.max() == 0:
            return []

        # Threshold at 40 % of the max distance to find "peak" regions.
        # A single compact robot gives one peak region; two touching robots
        # give two separate peak regions because there is a saddle point
        # between them where the distance drops below 40 %.
        _, peak_mask = cv2.threshold(
            dist, dist.max() * 0.40, 255, cv2.THRESH_BINARY)
        peak_mask = peak_mask.astype(np.uint8)

        # Erode to pull apart adjacent peak regions that still touch
        k_sep = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        peak_mask = cv2.erode(peak_mask, k_sep)

        # Each connected component of the peak mask = one robot center
        n_labels, labels = cv2.connectedComponents(peak_mask)
        centers = []
        for lbl in range(1, n_labels):
            comp = (labels == lbl).astype(np.uint8)
            M = cv2.moments(comp)
            if M["m00"] > 0:
                centers.append((M["m10"] / M["m00"],
                                 M["m01"] / M["m00"]))
        return centers

    def _get_blobs(self, frame):
        """
        Return up to 4 (fx_in, fy_in, heading_rad) tuples, sorted by
        blob area descending.

        Each foreground contour is passed through _split_contour() so that
        merged / touching robots produce multiple centers rather than one
        centroid halfway between them.
        """
        cv2, np = self.cv2, self.np

        diff = cv2.absdiff(frame, self._bg)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, fg = cv2.threshold(gray, self.FG_THRESH, 255, cv2.THRESH_BINARY)
        fg = cv2.bitwise_and(fg, self._field_mask)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kern)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self._kern)

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Collect all candidate centers with their parent-blob area
        candidates = []
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < self.BLOB_MIN:
                continue
            # Reject game balls: compute inscribed-circle radius via
            # distance transform. Balls ≤8 px radius; robots ≥11 px.
            _bm = np.zeros(self._field_mask.shape, dtype=np.uint8)
            cv2.drawContours(_bm, [c], -1, 255, -1)
            _dr = cv2.distanceTransform(_bm, cv2.DIST_L2, 5)
            if _dr.max() < self.MIN_RADIUS_PX:
                continue
            sub_centers = self._split_contour(c)
            # Divide area equally among sub-centers for ranking purposes
            per_center_area = area / max(len(sub_centers), 1)
            for (cx, cy) in sub_centers:
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                fp = cv2.perspectiveTransform(pt, self._H_2d)[0][0]
                fx, fy = float(fp[0]), float(fp[1])
                if not (0 <= fx <= 144 and 0 <= fy <= 144):
                    continue
                heading = 0.0  # heading from split center is unreliable
                candidates.append((fx, fy, heading, per_center_area))

        candidates.sort(key=lambda b: -b[3])
        return [(fx, fy, h) for fx, fy, h, _ in candidates[:4]]


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
    debug_every_n     = 10,
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
    if debug:
        print("[INFO] Saving debug frame every {} processed frames".format(
            debug_every_n))

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

            # Draw all foreground blobs and their split centers.
            # Yellow outline = robot-sized (passes radius filter)
            # Gray  outline = too small (ball or noise, filtered out)
            # Cyan cross    = split center fed to the tracker
            if tracker._bg is not None:
                all_cnts, split_centers = tracker._foreground_contours(frame)
                for c in all_cnts:
                    area = cv2.contourArea(c)
                    _bm2 = np.zeros(tracker._field_mask.shape, np.uint8)
                    cv2.drawContours(_bm2, [c], -1, 255, -1)
                    _dr2 = cv2.distanceTransform(_bm2, cv2.DIST_L2, 5)
                    is_robot = _dr2.max() >= tracker.MIN_RADIUS_PX
                    color = (0, 255, 255) if is_robot else (160, 160, 160)
                    cv2.drawContours(dbg, [c], -1, color, 2 if is_robot else 1)
                    if area >= tracker.BLOB_MIN:
                        M2 = cv2.moments(c)
                        if M2["m00"] > 0:
                            bx2 = int(M2["m10"] / M2["m00"])
                            by2 = int(M2["m01"] / M2["m00"])
                            cv2.putText(dbg, "{:.0f}".format(area),
                                        (bx2 - 15, by2 + 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                        color, 1)
                # Cyan crosses = split centers used by tracker
                for sx, sy in split_centers[:4]:
                    cv2.drawMarker(dbg, (sx, sy), (0, 255, 255),
                                   cv2.MARKER_CROSS, 16, 2)

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
            if processed % debug_every_n == 0:
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
    p.add_argument("--debug-every",  type=int, default=5,
                   help="Save 1 debug frame every N processed frames (default 5)")
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
        debug_every_n     = args.debug_every,
        manual_corners_px = corners,
    )


if __name__ == "__main__":
    main()