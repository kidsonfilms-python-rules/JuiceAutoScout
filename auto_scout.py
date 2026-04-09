# -*- coding: utf-8 -*-
"""
auto_scout.py — FTC DECODE robot tracker  (v3.2)

Base: v3.1 topology-based merge tracking.
Changes in v3.2 — fingerprint overhaul:

  a) DOMINANT COLOR CLASSIFIER — each robot blob is immediately classified
     into one of 4 perceptual color classes: WHITE, BLACK, COLORED, SILVER.
     Classification uses the V (brightness) and S (saturation) channels of the
     HSV median over the inner blob mask.  A white robot (high-V, low-S) gets a
     completely different color class from a black/silver robot — this is a hard
     discriminator that is reliable from frame 1, not a soft histogram cost that
     ramps up over 8+ updates.

  b) FULL HSV HISTOGRAM (H×S×V) — previously only H+S were used, ignoring
     brightness entirely.  White vs silver vs black are distinguished purely by
     V; adding it makes the histogram space much more discriminative.

  c) TIGHTER FINGERPRINT GATES —
       - Update threshold: d_fp < 0.45 (was 0.60)
       - Penalty threshold: d_fp > 0.50 (was 0.65)
       - Hard-reject threshold: d_fp > 0.80 → cost += 0.60 (new)
     The dead zone between update and penalize is now 0.05 units, not 0.15.

  d) FINGERPRINT ACTIVE FROM FRAME 1 — FP_MIN_UPDATES lowered to 3 (was 8).
     The dominant color class is available immediately even before the
     histogram stabilises, so the white robot is never mis-assigned.

  e) HIGHER FINGERPRINT WEIGHT CAP — raised from 0.60 to 0.85.  For a robot
     with a clearly distinctive color (white, bright orange, etc.) the color
     signal should be allowed to dominate the cost matrix.

  f) STREAK THRESHOLD RAISED to 3 (was 2) — reduces fingerprint corruption
     from brief occlusion or lighting transients.

  g) COLOR CLASS LOCK — once a track has accumulated ≥15 observations of the
     same color class, that class is "locked".  Any blob whose dominant class
     differs from the locked class pays a large additional cost (0.50), making
     cross-class swaps essentially impossible regardless of distance.

Usage:
  python3 auto_scout.py --no-download --video-path match.mp4 \\
      --corners field_corners.json [--debug] [--start-offset 1.0]

Dependencies:
  pip install opencv-python numpy progress
  pip install scipy          # optional — enables optimal assignment
"""

import argparse
import csv
import json
import math
import os
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from typing import Dict, FrozenSet, List, Optional, Tuple


def _require(package, pip_name=None):
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        name = pip_name or package
        print("[ERROR] Missing: {}\n  Install: pip install {}".format(name, name))
        sys.exit(1)


def _make_bar(label, max_val):
    try:
        from progress.bar import Bar
        return Bar(label, max=max_val,
                   suffix="%(percent).0f%% %(elapsed_td)s ETA %(eta_td)s")
    except ImportError:
        class _FallbackBar:
            def __init__(self, lbl, total):
                self._lbl   = lbl
                self._total = max(total, 1)
                self._n     = 0
                print("[{}] 0%".format(lbl), end="", flush=True)
            def next(self):
                self._n += 1
                pct = int(self._n / self._total * 100)
                if self._n % max(1, self._total // 20) == 0 or self._n == self._total:
                    print("\r[{}] {}%".format(self._lbl, pct), end="", flush=True)
            def finish(self):
                print()
        return _FallbackBar(label, max_val)


# ─────────────────────────────────────────────────────────────────────────────
# WPILOG writer
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
# MergeGroup
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MergeGroup:
    track_ids  : List[int]             = dc_field(default_factory=list)
    entry_axis : Tuple[float, float]   = (1.0, 0.0)
    parent_id  : int                   = -1
    crossed    : bool                  = False


# ─────────────────────────────────────────────────────────────────────────────
# Color classes
# ─────────────────────────────────────────────────────────────────────────────
# Each blob is classified into one of these perceptual buckets using
# the median HSV over its inner mask region.  The classification is
# fast, parameter-free, and immediately available on frame 1.
#
# Class boundaries (HSV space, OpenCV conventions: H∈[0,179], S/V∈[0,255]):
#
#   WHITE   : V ≥ 180  AND  S < 60
#   BLACK   : V < 70
#   COLORED : S ≥ 80   (hue-dominant — orange, red, blue, green, etc.)
#   SILVER  : everything else (mid-V, low-mid S — grey/metallic)
#
# These thresholds deliberately leave fuzzy gaps; a robot near a boundary
# is not locked until ≥ COLOR_LOCK_N observations agree.

COLOR_WHITE   = 0
COLOR_BLACK   = 1
COLOR_COLORED = 2
COLOR_SILVER  = 3
COLOR_UNKNOWN = 4

_COLOR_NAMES = {COLOR_WHITE: "WHITE", COLOR_BLACK: "BLACK",
                COLOR_COLORED: "COLORED", COLOR_SILVER: "SILVER",
                COLOR_UNKNOWN: "UNKNOWN"}


def classify_color(median_h, median_s, median_v):
    """Return COLOR_* constant for a blob given its inner-mask HSV medians."""
    if median_v >= 180 and median_s < 60:
        return COLOR_WHITE
    if median_v < 70:
        return COLOR_BLACK
    if median_s >= 80:
        return COLOR_COLORED
    return COLOR_SILVER


# ─────────────────────────────────────────────────────────────────────────────
# RobotFingerprint — per-track visual identity
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RobotFingerprint:
    """
    Stronger identity fingerprint (v3.2):
      - inner_hist  : full HSV histogram (H×S×V) inside robot body
      - ring_hist   : HSV histogram on outer ring / bumper edge
      - edge_hist   : angular edge-energy profile around centroid
      - shape_vec   : coarse shape stats
      - color_class : dominant perceptual color class (WHITE/BLACK/…)
      - color_votes : vote counts for each class (for lock detection)
    """
    inner_hist  : Optional[object] = None
    ring_hist   : Optional[object] = None
    edge_hist   : Optional[object] = None
    shape_vec   : Optional[object] = None
    color_class : int  = COLOR_UNKNOWN
    color_votes : object = None   # np.ndarray shape (5,), filled in on first update
    n_updates   : int  = 0

    # ── tunables ──────────────────────────────────────────────────────────
    FP_MIN_UPDATES  = 3      # was 8 — start influencing assignment immediately
    EMA_ALPHA       = 0.12
    H_BINS          = 12     # hue bins
    S_BINS          = 4      # saturation bins
    V_BINS          = 4      # brightness bins (NEW — distinguishes white/silver/black)
    EDGE_SECTORS    = 12
    COLOR_LOCK_N    = 15     # observations needed to lock color class
    COLOR_LOCK_FRAC = 0.70   # fraction of votes that must agree to lock


# ─────────────────────────────────────────────────────────────────────────────
# RobotTracker
# ─────────────────────────────────────────────────────────────────────────────
class RobotTracker:
    """
    Detect and track 4 robots using median background subtraction plus
    topology-aware merge handling.

    v3.2 improvements: see module docstring.
    """

    N_BG_SAMPLES  = 80
    FG_THRESH     = 25
    BLOB_MIN      = 300
    MIN_RADIUS_PX = 10
    KERNEL_PX     = 9
    MAX_COAST     = 60
    MAX_DIST_IN   = 30.0
    ROBOT_SIZE_IN = 18.0

    # ── fingerprint assignment gates (tightened vs v3.1) ─────────────────
    FP_UPDATE_MAX_DIST  = 0.45   # was 0.60 — only update when blob clearly matches
    FP_PENALTY_MIN_DIST = 0.50   # was 0.65 — penalise bad matches sooner
    FP_HARD_REJECT_DIST = 0.80   # NEW    — near-certain mismatch: large extra cost
    FP_PENALTY_COST     = 0.25   # cost added when dist > FP_PENALTY_MIN_DIST
    FP_HARD_REJECT_COST = 0.60   # cost added when dist > FP_HARD_REJECT_DIST
    FP_WEIGHT_CAP       = 0.85   # was 0.60 — allow color to dominate
    FP_STREAK_NEEDED    = 3      # was 2 — require more consecutive good frames

    # Cost added when a blob's color class conflicts with a locked track class
    COLOR_MISMATCH_COST = 0.50

    def __init__(self, cv2, np):
        self.cv2, self.np = cv2, np
        self._bg           = None
        self._field_mask   = None
        self._H_2d         = None
        self._H_inv        = None
        self._kern         = cv2.getStructuringElement(
                                 cv2.MORPH_ELLIPSE, (self.KERNEL_PX, self.KERNEL_PX))
        self._neck_kern    = None
        self._robot_max_px = 60

        self.tracked_poses  = [RobotPose() for _ in range(4)]
        self._pos           = [None] * 4
        self._pos_px        = [None] * 4
        self._coast         = [999]  * 4
        self._initialized   = False

        self._area_history  = [[] for _ in range(4)]
        self._fp_streak     = [0] * 4
        self._fingerprints  = [RobotFingerprint() for _ in range(4)]

        self._last_frame     = None
        self._last_fp_obs: Dict[int, object] = {}
        self._last_real_cost = None

        self._merge_groups: Dict[FrozenSet, MergeGroup] = {}

    # ── setup ────────────────────────────────────────────────────────────

    def setup(self, video_path, ordered_corners, frame_shape):
        cv2, np = self.cv2, self.np
        h, w = frame_shape[:2]

        tl, tr, br, bl = ordered_corners
        cx_poly = (tl[0]+tr[0]+br[0]+bl[0]) / 4
        cy_poly = (tl[1]+tr[1]+br[1]+bl[1]) / 4
        SIDE_PAD = 25; BOTTOM_PAD = 25; TOP_PAD = 5

        def _pad(x, y):
            dx = x - cx_poly; dy = y - cy_poly
            return (int(x + (SIDE_PAD   if dx > 0 else -SIDE_PAD)),
                    int(y + (BOTTOM_PAD if dy > 0 else -TOP_PAD)))

        poly = np.array([_pad(*tl), _pad(*tr), _pad(*br), _pad(*bl)], dtype=np.int32)
        self._field_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self._field_mask, [poly], 255)

        dst2d = np.array([[0,144],[144,144],[144,0],[0,0]], dtype=np.float32)
        self._H_2d, _ = cv2.findHomography(ordered_corners, dst2d)
        self._H_inv   = np.linalg.inv(self._H_2d)

        c_f = np.array([[[72.0, 72.0]]], dtype=np.float32)
        r_f = np.array([[[72.0 + self.ROBOT_SIZE_IN, 72.0]]], dtype=np.float32)
        u_f = np.array([[[72.0, 72.0 + self.ROBOT_SIZE_IN]]], dtype=np.float32)
        c_px = cv2.perspectiveTransform(c_f, self._H_inv)[0][0]
        r_px = cv2.perspectiveTransform(r_f, self._H_inv)[0][0]
        u_px = cv2.perspectiveTransform(u_f, self._H_inv)[0][0]
        px_x = math.hypot(r_px[0]-c_px[0], r_px[1]-c_px[1])
        px_y = math.hypot(u_px[0]-c_px[0], u_px[1]-c_px[1])
        self._robot_max_px = max(px_x, px_y)
        print("[INFO] Robot pixel footprint: {:.1f} px / 18 in".format(self._robot_max_px))

        neck_r = max(3, int(self._robot_max_px * 0.15))
        self._neck_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (neck_r*2+1, neck_r*2+1))
        print("[INFO] Neck-breaking kernel radius: {} px".format(neck_r))

        cap2 = cv2.VideoCapture(video_path)
        n_total = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        indices  = np.linspace(0, n_total - 1, self.N_BG_SAMPLES, dtype=int)
        bar = _make_bar("Building background", len(indices))
        frames = []
        for idx in indices:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, f = cap2.read()
            if ret:
                frames.append(f.astype(np.float32))
            bar.next()
        bar.finish()
        cap2.release()

        self._bg = (np.median(frames, axis=0).astype(np.uint8)
                    if frames else np.zeros((h, w, 3), dtype=np.uint8))
        print("[INFO] Background ready.")

    # ── main update ───────────────────────────────────────────────────────

    def update(self, frame, _H=None):
        cv2, np = self.cv2, self.np
        self._last_frame  = frame
        self._last_fp_obs = {}
        if self._bg is None or self._H_2d is None:
            for p in self.tracked_poses:
                p.visible = False
            return self.tracked_poses

        blobs = self._get_blobs(frame)

        if not self._initialized:
            if len(blobs) < 4:
                for p in self.tracked_poses:
                    p.visible = False
                return self.tracked_poses
            for i, b in enumerate(sorted(blobs[:4], key=lambda b: b[0])):
                fx, fy, hdg, _pid, cx, cy, _cnt = b
                self._pos[i]    = (fx, fy)
                self._pos_px[i] = (cx, cy)
                self._coast[i]  = 0
                self.tracked_poses[i].x_in    = fx
                self.tracked_poses[i].y_in    = fy
                self.tracked_poses[i].heading = hdg
                self.tracked_poses[i].visible = True
                init_area = float(cv2.contourArea(_cnt))
                self._area_history[i] = [init_area] * 5
                # Seed fingerprint including color class from frame 1
                fp_obs = self._build_fingerprint(frame, _cnt, cx, cy)
                if fp_obs is not None:
                    ih, rh, eh, sv, cc, cv_hsv = fp_obs
                    fp = self._fingerprints[i]
                    fp.inner_hist = ih.copy()
                    fp.ring_hist  = rh.copy()
                    fp.edge_hist  = eh.copy()
                    fp.shape_vec  = sv.copy()
                    fp.color_class = cc
                    fp.color_votes = np.zeros(5, dtype=np.float32)
                    fp.color_votes[cc] = RobotFingerprint.FP_MIN_UPDATES
                    fp.n_updates  = RobotFingerprint.FP_MIN_UPDATES
            self._initialized = True
            print("[INFO] Tracks initialized with color classes: {}".format(
                [_COLOR_NAMES[self._fingerprints[k].color_class] for k in range(4)]))
            return self.tracked_poses

        assignment = self._assign(blobs)

        pid_to_contour = {}
        for b in blobs:
            if b[3] not in pid_to_contour:
                pid_to_contour[b[3]] = b[6]

        pid_to_tracks = defaultdict(list)
        for ti, bi in assignment.items():
            pid_to_tracks[blobs[bi][3]].append(ti)
        merged_pids = {pid: tl for pid, tl in pid_to_tracks.items() if len(tl) > 1}

        active_keys = set()
        for pid, tlist in merged_pids.items():
            key = frozenset(tlist)
            active_keys.add(key)
            if key not in self._merge_groups:
                self._merge_groups[key] = self._create_merge_group(tlist, pid)
                print("[INFO] Merge started: tracks {}".format(sorted(tlist)))
            else:
                mg = self._merge_groups[key]
                mg.parent_id = pid
                if pid in pid_to_contour:
                    self._update_crossing(mg, pid_to_contour[pid])

        for key in list(self._merge_groups.keys()):
            if key not in active_keys:
                mg = self._merge_groups.pop(key)
                self._apply_separation(mg)

        merged_tracks = set()
        for mg in self._merge_groups.values():
            merged_tracks.update(mg.track_ids)

        for i in range(4):
            if i in assignment:
                fx, fy, hdg, pid, cx, cy, _cnt = blobs[assignment[i]]
                blob_area = float(cv2.contourArea(_cnt))
                self._coast[i] = 0
                self.tracked_poses[i].x_in    = fx
                self.tracked_poses[i].y_in    = fy
                self.tracked_poses[i].heading = hdg
                self.tracked_poses[i].visible = True
                if i not in merged_tracks:
                    self._pos[i]    = (fx, fy)
                    self._pos_px[i] = (cx, cy)
                    hist = self._area_history[i]
                    hist.append(blob_area)
                    if len(hist) > 20:
                        hist.pop(0)

                    bi = assignment[i]
                    rc = self._last_real_cost
                    assign_cost = float(rc[i, bi]) if (rc is not None and bi < rc.shape[1]) else 1.0
                    confident = assign_cost < 0.6

                    if confident and rc is not None and bi < rc.shape[1]:
                        for i2 in range(4):
                            if i2 != i and rc[i2, bi] - assign_cost < 0.15:
                                confident = False
                                break

                    if confident and len(hist) >= 5:
                        expected_a = float(self.np.median(hist[:-1]))
                        if expected_a > 0 and blob_area / expected_a > 2.5:
                            confident = False

                    if confident:
                        bi_key = assignment[i]
                        fp_obs = self._last_fp_obs.get(bi_key)
                        if fp_obs is None:
                            fp_obs = self._build_fingerprint(frame, _cnt, cx, cy)

                        if fp_obs is not None:
                            fp = self._fingerprints[i]
                            d_fp = self._fingerprint_distance(fp, fp_obs)

                            # Tightened streak gate
                            if d_fp < self.FP_UPDATE_MAX_DIST:
                                self._fp_streak[i] = min(self._fp_streak[i] + 1, 8)
                            else:
                                self._fp_streak[i] = 0

                            if self._fp_streak[i] >= self.FP_STREAK_NEEDED:
                                ih, rh, eh, sv, cc, cv_hsv = fp_obs
                                a = RobotFingerprint.EMA_ALPHA

                                if fp.inner_hist is None:
                                    fp.inner_hist  = ih.copy()
                                    fp.ring_hist   = rh.copy()
                                    fp.edge_hist   = eh.copy()
                                    fp.shape_vec   = sv.copy()
                                    fp.color_votes = np.zeros(5, dtype=np.float32)
                                else:
                                    fp.inner_hist  = (1 - a) * fp.inner_hist + a * ih
                                    fp.ring_hist   = (1 - a) * fp.ring_hist  + a * rh
                                    fp.edge_hist   = (1 - a) * fp.edge_hist  + a * eh
                                    fp.shape_vec   = (1 - a) * fp.shape_vec  + a * sv

                                # Update color vote tallies
                                if fp.color_votes is None:
                                    fp.color_votes = np.zeros(5, dtype=np.float32)
                                fp.color_votes[cc] += 1.0

                                # Re-derive dominant class from votes
                                total_votes = float(fp.color_votes.sum())
                                if total_votes >= 1:
                                    best_class = int(np.argmax(fp.color_votes))
                                    best_frac  = fp.color_votes[best_class] / total_votes
                                    if (total_votes >= RobotFingerprint.COLOR_LOCK_N
                                            and best_frac >= RobotFingerprint.COLOR_LOCK_FRAC):
                                        if fp.color_class != best_class:
                                            print("[INFO] Track {} color locked: {}".format(
                                                i, _COLOR_NAMES[best_class]))
                                        fp.color_class = best_class
                                    else:
                                        fp.color_class = best_class

                                fp.n_updates = min(fp.n_updates + 1, 500)
                else:
                    if self._pos[i] is not None:
                        ox, oy = self._pos[i]
                        self._pos[i] = (ox * 0.7 + fx * 0.3,
                                        oy * 0.7 + fy * 0.3)
            else:
                recovered = False
                if self._pos[i] is not None and self._bg is not None:
                    px_r, py_r = self._pos[i]
                    diff2 = cv2.absdiff(frame, self._bg)
                    gray2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2GRAY)
                    _, fg2 = cv2.threshold(gray2, max(8, self.FG_THRESH // 3),
                                           255, cv2.THRESH_BINARY)
                    fg2 = cv2.bitwise_and(fg2, self._field_mask)
                    fg2 = cv2.morphologyEx(fg2, cv2.MORPH_CLOSE, self._kern)
                    cnts2, _ = cv2.findContours(
                        fg2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    best_d2, best_blob = self.MAX_DIST_IN * 0.6, None
                    for c2 in cnts2:
                        if cv2.contourArea(c2) < self.BLOB_MIN // 2:
                            continue
                        M2 = cv2.moments(c2)
                        if M2["m00"] == 0:
                            continue
                        bx2 = M2["m10"] / M2["m00"]
                        by2 = M2["m01"] / M2["m00"]
                        pt2 = np.array([[[bx2, by2]]], dtype=np.float32)
                        fp2 = cv2.perspectiveTransform(pt2, self._H_2d)[0][0]
                        fx2, fy2 = float(fp2[0]), float(fp2[1])
                        if not (0 <= fx2 <= 144 and 0 <= fy2 <= 144):
                            continue
                        d2 = math.hypot(fx2 - px_r, fy2 - py_r)
                        if d2 < best_d2:
                            best_d2 = d2
                            best_blob = (fx2, fy2, int(bx2), int(by2))
                    if best_blob is not None:
                        fx2, fy2, cx2, cy2 = best_blob
                        self._pos[i]    = (fx2, fy2)
                        self._pos_px[i] = (cx2, cy2)
                        self.tracked_poses[i].x_in = fx2
                        self.tracked_poses[i].y_in = fy2
                        self.tracked_poses[i].visible = True
                        recovered = True

                if not recovered:
                    self._coast[i] = min(self._coast[i] + 1, self.MAX_COAST + 1)
                    self.tracked_poses[i].visible = (
                        self._coast[i] <= self.MAX_COAST
                        and self._pos[i] is not None)
                    if self._pos[i] is not None:
                        self.tracked_poses[i].x_in = self._pos[i][0]
                        self.tracked_poses[i].y_in = self._pos[i][1]

        return self.tracked_poses

    # ── optimal assignment ────────────────────────────────────────────────

    def _assign(self, blobs: list) -> Dict[int, int]:
        """
        Cost matrix with per-blob color class and full fingerprint.

        v3.2 additions:
          - Compute blob color class up-front for each blob.
          - Add COLOR_MISMATCH_COST when a blob's class conflicts with a
            locked track color class.  This makes cross-class swaps nearly
            impossible even when the robots are physically close.
          - Tighter fingerprint penalty/hard-reject thresholds.
          - Higher fp_weight cap (0.85).
        """
        np = self.np
        cv2 = self.cv2
        n = len(blobs)

        INF       = 1e9
        SKIP_COST = 2.0

        self._last_fp_obs: Dict[int, object] = {}

        # Pre-compute color class for each blob (cheap — just HSV median)
        blob_color_class = {}
        for j, b in enumerate(blobs):
            fp_obs = self._build_fingerprint_fast(b[6], b[4], b[5])
            if fp_obs is not None:
                blob_color_class[j] = fp_obs[4]   # index 4 = color_class
                self._last_fp_obs[j] = fp_obs
            else:
                blob_color_class[j] = COLOR_UNKNOWN

        real_cost = np.full((4, max(n, 1)), INF, dtype=np.float64)
        for i in range(4):
            if self._pos[i] is None:
                real_cost[i, :] = SKIP_COST * 0.9
                continue
            px, py = self._pos[i]
            hist = self._area_history[i]
            expected_area = float(np.median(hist)) if len(hist) >= 3 else None
            fp_i = self._fingerprints[i]
            fp_ready = fp_i.n_updates >= RobotFingerprint.FP_MIN_UPDATES

            # Is this track's color class locked?
            color_locked = (
                fp_i.color_class != COLOR_UNKNOWN
                and fp_i.color_votes is not None
                and float(fp_i.color_votes.sum()) >= RobotFingerprint.COLOR_LOCK_N
                and (float(fp_i.color_votes[fp_i.color_class])
                     / max(float(fp_i.color_votes.sum()), 1.0))
                     >= RobotFingerprint.COLOR_LOCK_FRAC
            )

            for j, b in enumerate(blobs):
                d = math.hypot(b[0]-px, b[1]-py)
                dist_cost = d / self.MAX_DIST_IN

                # Size mismatch penalty
                if expected_area is not None and expected_area > 0:
                    blob_area = float(cv2.contourArea(b[6]))
                    ratio = blob_area / expected_area
                    size_penalty = min(0.4, abs(math.log(max(ratio, 1e-3))) * 0.25)
                else:
                    size_penalty = 0.0

                # ── Color class mismatch penalty (NEW in v3.2) ────────────
                # If the track has a locked color class, penalise any blob
                # whose dominant class is different and not UNKNOWN.
                # This is the primary fix for the white-robot mis-assignment.
                bcc = blob_color_class.get(j, COLOR_UNKNOWN)
                color_cost = 0.0
                if (color_locked
                        and bcc != COLOR_UNKNOWN
                        and bcc != fp_i.color_class):
                    color_cost = self.COLOR_MISMATCH_COST

                # ── Fingerprint cost ──────────────────────────────────────
                fp_cost = 0.0
                if fp_ready:
                    # Weight grows from 0.20 → FP_WEIGHT_CAP as n_updates rises
                    fp_weight = min(
                        self.FP_WEIGHT_CAP,
                        0.20 + (fp_i.n_updates - RobotFingerprint.FP_MIN_UPDATES)
                               / 15.0 * (self.FP_WEIGHT_CAP - 0.20)
                    )
                    fp_obs = self._last_fp_obs.get(j)
                    if fp_obs is not None:
                        d_fp = self._fingerprint_distance(fp_i, fp_obs)
                        fp_cost = fp_weight * d_fp
                        if d_fp > self.FP_HARD_REJECT_DIST:
                            fp_cost += self.FP_HARD_REJECT_COST
                        elif d_fp > self.FP_PENALTY_MIN_DIST:
                            fp_cost += self.FP_PENALTY_COST

                real_cost[i, j] = dist_cost + size_penalty + color_cost + fp_cost

        if n == 0:
            return {}

        skip_cols = np.full((4, 4), INF, dtype=np.float64)
        for i in range(4):
            if self._pos[i] is not None:
                coast_frac = min(self._coast[i] / max(self.MAX_COAST, 1), 1.0)
                skip_cost_i = SKIP_COST * (1.0 - 0.75 * coast_frac)
            else:
                skip_cost_i = SKIP_COST * 0.9
            skip_cols[i, i] = skip_cost_i

        cost = np.hstack([real_cost, skip_cols])
        self._last_real_cost = real_cost

        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
            assignment = {}
            for r, c in zip(row_ind, col_ind):
                if c < n:
                    assignment[r] = c
        except ImportError:
            assignment = {}
            work = cost.copy()
            total_cols = n + 4
            for _ in range(4):
                idx = int(np.argmin(work))
                ri  = idx // total_cols
                ci  = idx  % total_cols
                if work[ri, ci] >= INF:
                    break
                if ci < n:
                    assignment[ri] = ci
                work[ri, :] = INF
                work[:, ci] = INF

        return assignment

    # ── merge group lifecycle ─────────────────────────────────────────────

    def _create_merge_group(self, track_ids: List[int], parent_id: int) -> MergeGroup:
        np = self.np
        positions = []
        for tid in track_ids:
            if self._pos_px[tid] is not None:
                positions.append((tid, float(self._pos_px[tid][0]),
                                       float(self._pos_px[tid][1])))
            elif self._pos[tid] is not None:
                pt = np.array([[[self._pos[tid][0], self._pos[tid][1]]]],
                              dtype=np.float32)
                ip = self.cv2.perspectiveTransform(pt, self._H_inv)[0][0]
                positions.append((tid, float(ip[0]), float(ip[1])))
            else:
                positions.append((tid, 0.0, 0.0))

        ax, ay = 1.0, 0.0
        best = 0.0
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dx = positions[j][1] - positions[i][1]
                dy = positions[j][2] - positions[i][2]
                d  = math.hypot(dx, dy)
                if d > best:
                    best = d
                    ax, ay = dx/max(d, 1e-9), dy/max(d, 1e-9)

        cx = sum(p[1] for p in positions) / len(positions)
        cy = sum(p[2] for p in positions) / len(positions)

        def _proj(px, py): return (px-cx)*ax + (py-cy)*ay
        ordered     = sorted(positions, key=lambda p: _proj(p[1], p[2]))
        ordered_ids = [p[0] for p in ordered]

        return MergeGroup(track_ids=ordered_ids, entry_axis=(ax, ay),
                          parent_id=parent_id, crossed=False)

    def _update_crossing(self, mg: MergeGroup, contour) -> None:
        cv2 = self.cv2
        peaks = self._split_contour(contour)
        if len(peaks) < 2:
            return
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return
        mcx = M["m10"] / M["m00"]
        mcy = M["m01"] / M["m00"]
        ax, ay = mg.entry_axis

        def _proj(px, py): return (px-mcx)*ax + (py-mcy)*ay
        p0, p1 = sorted(peaks[:2], key=lambda p: _proj(p[0], p[1]))
        dot = (p1[0]-p0[0])*ax + (p1[1]-p0[1])*ay
        mg.crossed = (dot < 0)

    def _apply_separation(self, mg: MergeGroup) -> None:
        """
        Resolve which track ID belongs to which exiting blob.

        Primary: fingerprint color class (if any robot in the group has a
                 locked class that differs from others, use it to disambiguate).
        Secondary: topology crossing flag.
        Tertiary: full fingerprint distance comparison.
        """
        tids = mg.track_ids
        n    = len(tids)

        # ── Color-class-based separation (NEW in v3.2) ────────────────────
        # If two tracks have different locked color classes, and we have two
        # blobs, assign the blob whose color matches each track's class.
        if (n == 2
                and self._last_frame is not None
                and self._last_real_cost is not None):
            fp0 = self._fingerprints[tids[0]]
            fp1 = self._fingerprints[tids[1]]
            cc0 = fp0.color_class
            cc1 = fp1.color_class
            if (cc0 != COLOR_UNKNOWN and cc1 != COLOR_UNKNOWN and cc0 != cc1):
                rc = self._last_real_cost
                n_blobs = rc.shape[1]
                best0 = min(range(n_blobs), key=lambda j: rc[tids[0], j]) if n_blobs else None
                best1 = min(range(n_blobs), key=lambda j: rc[tids[1], j]) if n_blobs else None
                if best0 is not None and best1 is not None and best0 != best1:
                    obs0 = self._last_fp_obs.get(best0)
                    obs1 = self._last_fp_obs.get(best1)
                    if obs0 is not None and obs1 is not None:
                        bcc0 = obs0[4]
                        bcc1 = obs1[4]
                        # If blob0 matches track1's class better, swap
                        match_a = (bcc0 == cc0 or bcc0 == COLOR_UNKNOWN) and \
                                  (bcc1 == cc1 or bcc1 == COLOR_UNKNOWN)
                        match_b = (bcc0 == cc1 or bcc0 == COLOR_UNKNOWN) and \
                                  (bcc1 == cc0 or bcc1 == COLOR_UNKNOWN)
                        if match_b and not match_a:
                            sp   = [self._pos[tids[k]]    for k in range(n)]
                            sppx = [self._pos_px[tids[k]] for k in range(n)]
                            for k in range(n):
                                self._pos[tids[k]]    = sp[n-1-k]
                                self._pos_px[tids[k]] = sppx[n-1-k]
                            print("[INFO] Separation swap (color-class) — tracks {}".format(
                                sorted(tids)))
                            return

        # ── Fingerprint-distance-based separation ─────────────────────────
        fp_swap = None
        if (n == 2
                and self._last_frame is not None
                and self._last_real_cost is not None):
            fp0 = self._fingerprints[tids[0]]
            fp1 = self._fingerprints[tids[1]]
            if (fp0.n_updates >= RobotFingerprint.FP_MIN_UPDATES * 2
                    and fp1.n_updates >= RobotFingerprint.FP_MIN_UPDATES * 2):
                rc = self._last_real_cost
                n_blobs = rc.shape[1]
                best0 = min(range(n_blobs), key=lambda j: rc[tids[0], j]) if n_blobs else None
                best1 = min(range(n_blobs), key=lambda j: rc[tids[1], j]) if n_blobs else None
                if best0 is not None and best1 is not None and best0 != best1:
                    obs0 = self._last_fp_obs.get(best0)
                    obs1 = self._last_fp_obs.get(best1)
                    if obs0 is not None and obs1 is not None:
                        score_a = (self._fingerprint_distance(fp0, obs0) +
                                   self._fingerprint_distance(fp1, obs1))
                        score_b = (self._fingerprint_distance(fp0, obs1) +
                                   self._fingerprint_distance(fp1, obs0))
                        fp_swap = score_b < score_a

        if fp_swap is not None:
            do_swap = fp_swap
            src = "fingerprint"
        else:
            do_swap = mg.crossed
            src = "topology"

        if do_swap:
            sp   = [self._pos[tids[k]]    for k in range(n)]
            sppx = [self._pos_px[tids[k]] for k in range(n)]
            for k in range(n):
                self._pos[tids[k]]    = sp[n-1-k]
                self._pos_px[tids[k]] = sppx[n-1-k]
            print("[INFO] Separation swap ({}) — tracks {}".format(src, sorted(tids)))
        else:
            print("[INFO] Separation no-swap ({}) — tracks {}".format(src, sorted(tids)))

    # ── fingerprint construction ──────────────────────────────────────────

    def _build_fingerprint(self, frame, contour, cx_px, cy_px):
        """
        Build fingerprint from full HSV histogram (H×S×V, v3.2).

        Returns (inner_hist, ring_hist, edge_hist, shape_vec, color_class,
                 (median_h, median_s, median_v))
        or None.
        """
        cv2, np = self.cv2, self.np
        h_fr, w_fr = frame.shape[:2]

        x, y, bw, bh = cv2.boundingRect(contour)
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_fr, x + bw + pad)
        y2 = min(h_fr, y + bh + pad)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        crop = frame[y1:y2, x1:x2].copy()

        mask_full = np.zeros((h_fr, w_fr), dtype=np.uint8)
        cv2.drawContours(mask_full, [contour], -1, 255, -1)
        mask_crop = mask_full[y1:y2, x1:x2]

        inner_r = max(2, int(min(bw, bh) * 0.08))
        inner_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (inner_r * 2 + 1, inner_r * 2 + 1))
        inner_mask = cv2.erode(mask_crop, inner_k)
        ring_mask = cv2.subtract(mask_crop, inner_mask)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        H = RobotFingerprint.H_BINS
        S = RobotFingerprint.S_BINS
        V = RobotFingerprint.V_BINS   # NEW dimension

        def _hist3(mask):
            """Full H×S×V histogram, normalised."""
            hst = cv2.calcHist(
                [hsv], [0, 1, 2], mask,
                [H, S, V],
                [0, 180, 0, 256, 0, 256]
            ).flatten().astype(np.float32)
            s = float(hst.sum())
            if s < 1e-6:
                return None
            return hst / s

        inner_hist = _hist3(inner_mask)
        ring_hist  = _hist3(ring_mask)
        if inner_hist is None:
            return None
        if ring_hist is None:
            ring_hist = inner_hist.copy()

        # ── Dominant color classification from inner mask HSV medians ─────
        hsv_inner = hsv[inner_mask > 0]   # shape (N, 3)
        if len(hsv_inner) < 5:
            # Fallback to full mask
            hsv_inner = hsv[mask_crop > 0]
        if len(hsv_inner) < 5:
            return None

        med_h = float(np.median(hsv_inner[:, 0]))
        med_s = float(np.median(hsv_inner[:, 1]))
        med_v = float(np.median(hsv_inner[:, 2]))
        color_class = classify_color(med_h, med_s, med_v)

        # Edge-energy by angular sector
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)
        mag[mask_crop == 0] = 0

        yy, xx = np.indices(gray.shape)
        cx_loc = float(cx_px - x1)
        cy_loc = float(cy_px - y1)
        ang = np.arctan2(yy - cy_loc, xx - cx_loc)

        N = RobotFingerprint.EDGE_SECTORS
        eh = np.zeros(N, dtype=np.float32)
        sector_w = 2 * math.pi / N
        for k in range(N):
            lo = -math.pi + k * sector_w
            hi = lo + sector_w
            in_sector = (ang >= lo) & (ang < hi if k < N-1 else ang <= hi)
            eh[k] = float(mag[in_sector].sum())

        es = float(eh.sum())
        if es < 1e-6:
            eh[:] = 1.0 / N
        else:
            eh /= es

        area = float(cv2.contourArea(contour))
        rect_area = max(float(bw * bh), 1.0)
        hull = cv2.convexHull(contour)
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        solidity = max(0.0, min(1.0, area / hull_area))
        extent   = max(0.0, min(1.0, area / rect_area))
        aspect   = max(bw, bh) / max(min(bw, bh), 1)
        aspect   = max(0.0, min(1.0, aspect / 3.0))

        hu = cv2.HuMoments(cv2.moments(contour)).flatten()
        hu = np.nan_to_num(-np.sign(hu) * np.log10(np.abs(hu) + 1e-12),
                           nan=0.0, posinf=0.0, neginf=0.0)
        hu = np.clip(hu[:3] / 10.0, -1.0, 1.0).astype(np.float32)
        shape_vec = np.array([solidity, extent, aspect,
                               hu[0], hu[1], hu[2]], dtype=np.float32)

        return inner_hist, ring_hist, eh, shape_vec, color_class, (med_h, med_s, med_v)

    def _build_fingerprint_fast(self, contour, cx_px, cy_px):
        if self._last_frame is None:
            return None
        return self._build_fingerprint(self._last_frame, contour, cx_px, cy_px)

    @staticmethod
    def _bhattacharyya(h1, h2):
        import numpy as np_mod
        if h1 is None or h2 is None:
            return 0.5
        coeff = float(np_mod.sum(np_mod.sqrt(np_mod.clip(h1 * h2, 0, None))))
        coeff = max(min(coeff, 1.0), 1e-9)
        return min(1.0, -math.log(coeff) / 3.0)

    def _circular_bhattacharyya(self, h1, h2):
        if h1 is None or h2 is None:
            return 0.5
        best = 1.0
        for s in range(len(h2)):
            d = self._bhattacharyya(h1, self.np.roll(h2, s))
            if d < best:
                best = d
        return best

    def _fingerprint_distance(self, fp, obs):
        """
        obs = (inner_hist, ring_hist, edge_hist, shape_vec, color_class, hsv_medians)
        Lower is better.
        """
        if fp is None or obs is None:
            return 0.75

        ih, rh, eh, sv = obs[0], obs[1], obs[2], obs[3]
        parts = []
        weights = []

        if fp.inner_hist is not None and ih is not None:
            parts.append(self._bhattacharyya(fp.inner_hist, ih))
            weights.append(0.35)

        if fp.ring_hist is not None and rh is not None:
            parts.append(self._bhattacharyya(fp.ring_hist, rh))
            weights.append(0.35)

        if fp.edge_hist is not None and eh is not None:
            parts.append(self._circular_bhattacharyya(fp.edge_hist, eh))
            weights.append(0.20)

        if fp.shape_vec is not None and sv is not None:
            d = float(self.np.linalg.norm(fp.shape_vec - sv))
            parts.append(min(1.0, d / 1.5))
            weights.append(0.10)

        if not parts:
            return 0.75

        return float(sum(p * w for p, w in zip(parts, weights)) / sum(weights))

    # ── blob splitting / watershed ────────────────────────────────────────

    def _split_contour(self, contour, seed_positions_px=None):
        cv2, np = self.cv2, self.np
        h, w = self._field_mask.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist.max() == 0:
            return []

        _, peak_mask = cv2.threshold(dist, dist.max() * 0.40, 255, cv2.THRESH_BINARY)
        peak_mask = peak_mask.astype(np.uint8)
        k_sep = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        peak_mask = cv2.erode(peak_mask, k_sep)
        n_labels, labels = cv2.connectedComponents(peak_mask)

        centers = []
        for lbl in range(1, n_labels):
            comp = (labels == lbl).astype(np.uint8)
            M = cv2.moments(comp)
            if M["m00"] > 0:
                centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))

        if (len(centers) < 2
                and seed_positions_px is not None
                and len(seed_positions_px) == 2
                and self._robot_max_px > 0):
            x, y, bw, bh = cv2.boundingRect(contour)
            if max(bw, bh) >= self._robot_max_px * 1.4:
                ws_centers = self._watershed_split(mask, seed_positions_px)
                if len(ws_centers) == 2:
                    return ws_centers

        return centers

    def _watershed_split(self, binary_mask, seed_positions_px):
        cv2, np = self.cv2, self.np
        h, w = binary_mask.shape[:2]
        img3 = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        markers = np.zeros((h, w), dtype=np.int32)
        seed_r  = max(4, int(self._robot_max_px * 0.1))
        for lbl, (sx, sy) in enumerate(seed_positions_px, start=1):
            sx, sy = int(sx), int(sy)
            if 0 <= sx < w and 0 <= sy < h:
                cv2.circle(markers, (sx, sy), seed_r, lbl, -1)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(binary_mask, kernel, iterations=2)
        outer   = cv2.bitwise_and(dilated, cv2.bitwise_not(binary_mask))
        markers[outer > 0] = 3
        if not (np.any(markers == 1) and np.any(markers == 2)):
            return []
        cv2.watershed(img3, markers)
        centers = []
        for lbl in [1, 2]:
            region = (markers == lbl).astype(np.uint8)
            M = cv2.moments(region)
            if M["m00"] > 0:
                centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        return centers if len(centers) == 2 else []

    # ── foreground mask ───────────────────────────────────────────────────

    def _foreground_mask(self, frame):
        cv2, np = self.cv2, self.np
        diff = cv2.absdiff(frame, self._bg)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, fg = cv2.threshold(gray, self.FG_THRESH, 255, cv2.THRESH_BINARY)
        fg = cv2.bitwise_and(fg, self._field_mask)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kern)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self._kern)
        if self._neck_kern is not None:
            fg = cv2.erode(fg,  self._neck_kern)
            fg = cv2.dilate(fg, self._neck_kern)
        return fg

    def _foreground_contours(self, frame):
        cv2, np = self.cv2, self.np
        fg = self._foreground_mask(frame)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = sorted([c for c in cnts if cv2.contourArea(c) >= self.BLOB_MIN],
                       key=lambda c: -cv2.contourArea(c))
        all_split = []
        for c in valid:
            area = float(cv2.contourArea(c))
            subs = self._split_contour(c)
            per  = area / max(len(subs), 1)
            for sx, sy in subs:
                all_split.append((sx, sy, per))
        all_split.sort(key=lambda x: -x[2])
        return valid, [(int(sx), int(sy)) for sx, sy, _ in all_split]

    def _get_blobs(self, frame):
        cv2, np = self.cv2, self.np
        fg = self._foreground_mask(frame)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for parent_id, c in enumerate(cnts):
            area = float(cv2.contourArea(c))
            if area < self.BLOB_MIN:
                continue
            _bm = np.zeros(self._field_mask.shape, dtype=np.uint8)
            cv2.drawContours(_bm, [c], -1, 255, -1)
            _dr = cv2.distanceTransform(_bm, cv2.DIST_L2, 5)
            if _dr.max() < self.MIN_RADIUS_PX:
                continue

            seeds = None
            x, y, bw, bh = cv2.boundingRect(c)
            if max(bw, bh) >= self._robot_max_px * 1.4 and self._initialized:
                seeds = []
                for tid in range(4):
                    if self._pos_px[tid] is None:
                        continue
                    px_t, py_t = self._pos_px[tid]
                    if cv2.pointPolygonTest(c, (float(px_t), float(py_t)), False) >= 0:
                        seeds.append((px_t, py_t))
                if len(seeds) < 2:
                    seeds = None

            sub_centers = self._split_contour(c, seed_positions_px=seeds)
            if not sub_centers:
                continue
            per_area = area / len(sub_centers)
            for cx, cy in sub_centers:
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                fp = cv2.perspectiveTransform(pt, self._H_2d)[0][0]
                fx, fy = float(fp[0]), float(fp[1])
                if not (0 <= fx <= 144 and 0 <= fy <= 144):
                    continue
                candidates.append((fx, fy, 0.0, per_area, parent_id,
                                    int(cx), int(cy), c))

        candidates.sort(key=lambda b: -b[3])
        return [(fx, fy, hdg, pid, cx, cy, cnt)
                for fx, fy, hdg, _a, pid, cx, cy, cnt in candidates]


# ─────────────────────────────────────────────────────────────────────────────
# Field corner auto-detector
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
        print("[INFO] Saving debug frame every {} processed frames".format(debug_every_n))

    csv_path    = os.path.join(output_dir, "robot_positions.csv")
    wpilog_path = os.path.join(output_dir, "match_log.wpilog")
    debug_dir   = os.path.join(output_dir, "tracker_debug") if debug else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    log = WPILogWriter(wpilog_path)
    pose_eids = [log.start_entry("Robot{}/Pose".format(i),    "double[]") for i in range(4)]
    vis_eids  = [log.start_entry("Robot{}/Visible".format(i), "boolean")  for i in range(4)]

    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp_s",
        "robot0_x_in","robot0_y_in","robot0_heading_rad","robot0_visible",
        "robot1_x_in","robot1_y_in","robot1_heading_rad","robot1_visible",
        "robot2_x_in","robot2_y_in","robot2_heading_rad","robot2_visible",
        "robot3_x_in","robot3_y_in","robot3_heading_rad","robot3_visible",
    ])

    tracker        = RobotTracker(cv2, np)
    field_detector = FieldDetector(cv2, np)

    ordered = None; H_2d = None; H_inv = None

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
        if tracker._bg is not None:
            bg_path = os.path.join(output_dir, "median_background.jpg")
            cv2.imwrite(bg_path, tracker._bg)
            print("[INFO] Median background saved: {}".format(bg_path))
    else:
        print("[WARN] No corners — will attempt auto-detection.")

    frame_num = start_frame; processed = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame if start_frame > 0 else 0)

    debug_colors = [(220,160,0),(0,220,255),(0,0,220),(0,120,255)]
    frames_to_process = max(1, (total_frames-start_frame+frame_step-1)//frame_step)
    bar = _make_bar("Processing frames", frames_to_process)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        match_time_s = frame_num / video_fps - start_offset_sec
        timestamp_us = max(0, int(match_time_s * 1_000_000))
        frame_num   += 1

        if (frame_num - start_frame) % frame_step != 0:
            continue

        bar.next()

        if ordered is None:
            result = field_detector.detect_field(frame)
            if result is not None:
                ordered, H_2d = result
                H_inv = np.linalg.inv(H_2d)
                tracker.setup(video_path, ordered, frame.shape)
                print("\n  [t={:.1f}s] Field auto-detected.".format(match_time_s))
                if tracker._bg is not None:
                    cv2.imwrite(os.path.join(output_dir, "median_background.jpg"),
                                tracker._bg)

        poses = tracker.update(frame)

        for i, p in enumerate(poses):
            log.write_pose2d(pose_eids[i], timestamp_us, p.x_m, p.y_m, p.heading)
            log.write_boolean(vis_eids[i], timestamp_us, p.visible)

        row = ["{:.4f}".format(match_time_s)]
        for p in poses:
            row += ["{:.2f}".format(p.x_in), "{:.2f}".format(p.y_in),
                    "{:.4f}".format(p.heading), "1" if p.visible else "0"]
        csv_writer.writerow(row)

        if debug and debug_dir and H_inv is not None:
            dbg = frame.copy()
            if ordered is not None:
                cv2.polylines(dbg, [ordered.astype(np.int32).reshape(-1,1,2)],
                              True, (0, 200, 0), 2)

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
                                        (bx2-15, by2+4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                for sx, sy in split_centers[:4]:
                    cv2.drawMarker(dbg, (sx, sy), (0, 255, 255),
                                   cv2.MARKER_CROSS, 16, 2)

            fh_d, fw_d = frame.shape[:2]
            cv2.circle(dbg, (fw_d-40, 40),
                       int(tracker._robot_max_px / 2), (80, 80, 80), 1)
            cv2.putText(dbg, "18in", (fw_d-74, 57),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1)

            # Merge group annotations
            for mg in tracker._merge_groups.values():
                pos_list = [tracker._pos_px[t] for t in mg.track_ids
                            if tracker._pos_px[t] is not None]
                if pos_list:
                    mcx = int(sum(p[0] for p in pos_list) / len(pos_list))
                    mcy = int(sum(p[1] for p in pos_list) / len(pos_list))
                    ax, ay = mg.entry_axis
                    ex = int(mcx+ax*50); ey = int(mcy+ay*50)
                    col = (0, 80, 255) if mg.crossed else (0, 220, 100)
                    cv2.arrowedLine(dbg, (mcx, mcy), (ex, ey), col, 2)
                    lbl2 = "+".join("R{}".format(t) for t in mg.track_ids)
                    lbl2 += " CROSSED" if mg.crossed else " ok"
                    cv2.putText(dbg, lbl2, (mcx+6, mcy-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

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
                # Show color class in debug label
                cc_name = _COLOR_NAMES.get(tracker._fingerprints[i].color_class, "?")
                lbl_txt = "R{} {}".format(i, cc_name[:3])
                cv2.putText(dbg, lbl_txt, (ix-10, iy-24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                dx = int(28 * math.cos(p.heading))
                dy = int(-28 * math.sin(p.heading))
                cv2.arrowedLine(dbg, (ix, iy), (ix+dx, iy+dy), (255,255,255), 2)
                n_drawn += 1

            n_merged = sum(len(mg.track_ids) for mg in tracker._merge_groups.values())
            lbl = ("init" if not tracker._initialized else
                   "{}/4 ({} merged)".format(n_drawn, n_merged) if n_merged else
                   "{}/4".format(n_drawn))
            cv2.putText(dbg, "t={:.2f}s  [{}]".format(match_time_s, lbl),
                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            if processed % debug_every_n == 0:
                cv2.imwrite(os.path.join(debug_dir,
                            "frame_{:06d}.jpg".format(processed)), dbg)

        processed += 1

    bar.finish()
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
|  5. Click icon LEFT of the field name ->                         |
|       Format: Pose2d   Units: Meters + Radians                   |
|  6. Repeat for Robot1, Robot2, Robot3                            |
|  7. Press play!                                                  |
|                                                                  |
|  Robot IDs: assigned left-to-right at match start.               |
|  Debug frames show color class (WHI/BLK/COL/SIL) per track,     |
|  merge axis + CROSSED/ok, and 18" ref circle.                    |
+------------------------------------------------------------------+
""")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def download_video(url, output_dir):
    import subprocess
    out = os.path.join(output_dir, "match_video.mp4")
    print("[INFO] Downloading:", url)
    bar = _make_bar("Downloading", 100)
    last_pct = 0
    proc = subprocess.Popen(
        ["yt-dlp", "-f", "best[ext=mp4]", "-o", out, url,
         "--progress-template", "%(progress._percent_str)s"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.stdout:
        for line in proc.stdout:
            try:
                pct = int(float(line.strip().rstrip("%")))
                while last_pct < pct:
                    bar.next(); last_pct += 1
            except ValueError:
                pass
    proc.wait()
    while last_pct < 100:
        bar.next(); last_pct += 1
    bar.finish()
    if proc.returncode != 0:
        print("[ERROR] yt-dlp failed"); sys.exit(1)
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