# -*- coding: utf-8 -*-
"""
auto_scout.py — FTC DECODE robot tracker  (v3.2)

Base: v3.1 topology-based merge tracking.
Changes in v3.2 — multi-robot merge fixes only:

  PROBLEM: When 3 or 4 robots merge and rearrange while overlapping, the
  tracker incorrectly reassigns IDs after separation because:
    (a) peak_assignment used predicted positions that drift away from the
        blob centroid during long merges (velocity extrapolation on frozen tracks)
    (b) no permutation/crossing tracking for 3+ merges (only 2-robot case
        had the "crossed" flip)
    (c) if the dist-transform peak extractor returned <expected peaks for
        even one frame, peak_assignment went stale without a fallback
    (d) after separation, re-anchoring used the last frame's peak positions
        which are the exit positions of the blob boundary — not where the
        robots actually are heading

  FIXES:

  e) PEAK-ANCHORED POSITION during merge — while 3+ robots are merged,
     their in-merge position is continuously updated to the nearest dist-
     transform peak (like _update_crossing already does for 3+), so velocity
     and position state track the peak rather than freezing at entry.
     This makes _predict_image_pos work correctly during the merge.

  f) PERMUTATION TRACKING for 3+ merges — MergeGroup now stores
     `entry_order` (the sorted track IDs at merge-start) and `current_order`
     (updated every frame from peak assignments sorted along entry_axis).
     On separation, the mapping entry_order[k] → current_order[k] gives the
     correct post-merge ID assignment, analogous to `crossed` for 2-robots.

  g) STALE PEAK GUARD — if _split_contour returns fewer peaks than tracks in
     the merge group, we fall back to the last good peak_assignment rather
     than silently doing nothing. This prevents the permutation state from
     freezing mid-rearrangement.

  h) SEPARATION RE-ANCHOR uses the permutation map to swap track state
     (pos, vel, pos_px, vel_px) so each logical robot ID ends up owning the
     correct physical position after the merge resolves.
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

SAVE_ALL_DEBUG_AROUND_MERGES = False
PROCESS_EVERY_SOURCE_FRAME = True


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
# MergeGroup  (v3.2: adds entry_order / current_order for permutation tracking)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MergeGroup:
    """
    State for 2 (or more) tracks sharing a single foreground blob.

    track_ids       : tracks in the group (unordered set, kept as list).
    entry_axis      : unit vector (ax, ay) in IMAGE pixel space pointing from
                      the "low" end to the "high" end at merge time.
    parent_id       : contour parent_id of the current merged blob.

    -- 2-robot fields (unchanged from v3.1) --
    crossed         : True if the two robots have swapped sides of entry_axis.

    -- 3+ robot fields (new in v3.2) --
    entry_order     : track IDs sorted by projection onto entry_axis at the
                      moment the merge was created.  Immutable after creation.
                      entry_order[0] is the robot that was most "negative"
                      along entry_axis at merge start.
    current_order   : track IDs sorted by projection of their most-recent
                      dist-transform peak onto entry_axis.  Updated every
                      frame.  At separation, entry_order[k]→current_order[k]
                      gives the permutation to apply.
    peak_assignment : {track_id: (px, py)} — current dist-transform peak per
                      track.  Used both for live position updates (fix e) and
                      for re-anchoring on separation (fix h).
    """
    track_ids      : List[int]             = dc_field(default_factory=list)
    entry_axis     : Tuple[float, float]   = (1.0, 0.0)
    parent_id      : int                   = -1
    # 2-robot
    crossed        : bool                  = False
    # 3+ robot (v3.2)
    entry_order    : List[int]             = dc_field(default_factory=list)
    current_order  : List[int]             = dc_field(default_factory=list)
    peak_assignment: dict                  = dc_field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# RobotTracker
# ─────────────────────────────────────────────────────────────────────────────
class RobotTracker:
    N_BG_SAMPLES  = 80
    FG_THRESH     = 30       # Raised from 22: reduces noise from crowd/alliance members
    BLOB_MIN      = 350      # Raised from 260: skip tiny noise blobs
    MIN_RADIUS_PX = 6        # Lowered from 8: far-wall robots appear smaller
    KERNEL_PX     = 7        # Lowered from 9: better for 640x360 video resolution
    MAX_COAST     = 60
    MAX_DIST_IN   = 30.0
    ROBOT_SIZE_IN = 18.0
    MAX_SPEED_IN  = 140.0    # Raised from 120: allow slightly faster apparent motion
    MAX_REACQ_IN  = 144.0
    REACQ_PX_PAD  = 2.0      # Raised from 1.6: wider reacquisition window
    VISIBLE_COAST = 8
    MERGE_HOLD    = 16
    MERGE_DEBUG_HOLD = 0
    EXPECT_2WAY_AREA = 1.6
    EXPECT_3WAY_AREA = 2.6
    EXPECT_4WAY_AREA = 3.5
    RELAXED_PEAK_RATIO_MULTI = 0.30
    RELAXED_PEAK_RATIO_PAIR  = 0.35
    RELAXED_SEP_MULTI = 3
    RELAXED_SEP_PAIR  = 5
    MERGE_PRIOR_PAD_PX = 0.75
    MERGE_PRIOR_MIN_SEP = 0.35
    BLOB_MIN_FIELD_AREA_IN2 = 40.0  # Lowered from 55: more permissive for far-wall robots
    BALL_FILTER_ENABLED = True
    BALL_GREEN_HSV_LO = (50, 95, 80)
    BALL_GREEN_HSV_HI = (100, 255, 255)
    BALL_PURPLE_HSV_LO = (132, 80, 70)
    BALL_PURPLE_HSV_HI = (165, 255, 255)
    BALL_MIN_FIELD_AREA_IN2 = 5.0
    BALL_MASK_OPEN_FRAC = 0.030
    BALL_MASK_CLOSE_FRAC = 0.055
    BALL_MASK_DILATE_FRAC = 0.070

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

        self.tracked_poses = [RobotPose() for _ in range(4)]
        self._pos          = [None] * 4
        self._pos_px       = [None] * 4
        self._vel          = [(0.0, 0.0)] * 4
        self._vel_px       = [(0.0, 0.0)] * 4
        self._coast        = [999]  * 4
        self._merge_recent = [0]    * 4
        self._initialized  = False

        self._merge_groups: Dict[FrozenSet, MergeGroup] = {}
        self._underresolved_tracks = set()
        self._last_ball_mask = None

    # ── setup ────────────────────────────────────────────────────────────

    def setup(self, video_path, ordered_corners, frame_shape):
        cv2, np = self.cv2, self.np
        h, w = frame_shape[:2]

        tl, tr, br, bl = ordered_corners
        cx_poly = (tl[0]+tr[0]+br[0]+bl[0]) / 4
        cy_poly = (tl[1]+tr[1]+br[1]+bl[1]) / 4
        # Tightened SIDE_PAD (10 vs 25) to reduce crowd/alliance-member bleed-in.
        # TOP_PAD increased (12 vs 5) to capture robots right at the far wall.
        # BOTTOM_PAD reduced (18 vs 25) to trim scoreboard noise.
        SIDE_PAD = 10; BOTTOM_PAD = 18; TOP_PAD = 12

        def _pad(x, y):
            dx = x - cx_poly; dy = y - cy_poly
            return (int(x + (SIDE_PAD   if dx > 0 else -SIDE_PAD)),
                    int(y + (BOTTOM_PAD if dy > 0 else -TOP_PAD)))

        poly = np.array([_pad(*tl), _pad(*tr), _pad(*br), _pad(*bl)], dtype=np.int32)
        self._field_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self._field_mask, [poly], 255)

        dst2d = np.array([[0,0],[144,0],[144,144],[0,144]], dtype=np.float32)
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
        if self._bg is None or self._H_2d is None:
            for p in self.tracked_poses:
                p.visible = False
            return self.tracked_poses

        for i in range(4):
            if self._merge_recent[i] > 0:
                self._merge_recent[i] -= 1

        blobs = self._get_blobs(frame)
        self._underresolved_tracks = self._find_underresolved_tracks(blobs)

        if not self._initialized:
            if len(blobs) < 4:
                for p in self.tracked_poses:
                    p.visible = False
                return self.tracked_poses
            for i, b in enumerate(sorted(blobs[:4], key=lambda b: b[0])):
                fx, fy, hdg, _pid, cx, cy, _qual, _nsplit, _cnt, *_rest = b
                self._pos[i]    = (fx, fy)
                self._pos_px[i] = (cx, cy)
                self._coast[i]  = 0
                self.tracked_poses[i].x_in    = fx
                self.tracked_poses[i].y_in    = fy
                self.tracked_poses[i].heading = hdg
                self.tracked_poses[i].visible = True
            self._initialized = True
            return self.tracked_poses

        assignment = self._assign(blobs)

        pid_to_contour = {}
        for b in blobs:
            if b[3] not in pid_to_contour:
                pid_to_contour[b[3]] = b[8]

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
            mg = self._merge_groups[key]
            mg.parent_id = pid
            if pid in pid_to_contour:
                self._update_crossing(mg, pid_to_contour[pid])
            for tid in tlist:
                self._merge_recent[tid] = self.MERGE_HOLD

        for key in list(self._merge_groups.keys()):
            if key not in active_keys:
                mg = self._merge_groups.pop(key)
                self._apply_separation(mg)

        # tracks inside a 2-robot merge have positions frozen (handled below)
        # tracks inside a 3+ robot merge get positions from peak_assignment
        two_merged_tracks = set()
        multi_merged_peaks = {}
        for mg in self._merge_groups.values():
            if len(mg.track_ids) == 2:
                two_merged_tracks.update(mg.track_ids)
            else:
                # FIX (e): use continuously-updated peak positions for 3+ merges
                multi_merged_peaks.update(mg.peak_assignment)

        for i in range(4):
            if i in multi_merged_peaks:
                px, py = multi_merged_peaks[i]
                pt = self.np.array([[[px, py]]], dtype=self.np.float32)
                fp = self.cv2.perspectiveTransform(pt, self._H_2d)[0][0]
                fx, fy = float(fp[0]), float(fp[1])
                self._coast[i] = 0
                self.tracked_poses[i].x_in = fx
                self.tracked_poses[i].y_in = fy
                self.tracked_poses[i].visible = True
                self._update_track_motion(i, (fx, fy), (px, py))
                self.tracked_poses[i].heading = self._motion_heading(
                    i, self.tracked_poses[i].heading)
                self._pos[i] = (fx, fy)
                self._pos_px[i] = (px, py)
                continue
            if i in assignment:
                fx, fy, _hdg, _pid, cx, cy, _qual, _nsplit, _cnt, *_rest = blobs[assignment[i]]
                self._coast[i] = 0
                self.tracked_poses[i].x_in    = fx
                self.tracked_poses[i].y_in    = fy
                self.tracked_poses[i].visible = True
                if i not in two_merged_tracks:
                    self._update_track_motion(i, (fx, fy), (cx, cy))
                    self.tracked_poses[i].heading = self._motion_heading(
                        i, self.tracked_poses[i].heading)
                    self._pos[i]    = (fx, fy)
                    self._pos_px[i] = (cx, cy)
                else:
                    self.tracked_poses[i].heading = self._motion_heading(
                        i, self.tracked_poses[i].heading)
            else:
                self._coast[i] = min(self._coast[i] + 1, self.MAX_COAST + 1)
                self.tracked_poses[i].visible = (
                    self._coast[i] <= self.VISIBLE_COAST
                    and self._pos[i] is not None)
                if self._pos[i] is not None:
                    self.tracked_poses[i].x_in = self._pos[i][0]
                    self.tracked_poses[i].y_in = self._pos[i][1]
                    self.tracked_poses[i].heading = self._motion_heading(
                        i, self.tracked_poses[i].heading)

        return self.tracked_poses

    # ── optimal assignment ────────────────────────────────────────────────

    def _assign(self, blobs: list) -> Dict[int, int]:
        np = self.np
        n = len(blobs)

        INF       = 1e9
        SKIP_COST = 2.0

        real_cost = np.full((4, max(n, 1)), INF, dtype=np.float64)
        for i in range(4):
            if self._pos[i] is None:
                real_cost[i, :] = SKIP_COST * 0.9
                continue
            pred_x, pred_y = self._predict_field_pos(i)
            pred_px_x, pred_px_y = self._predict_image_pos(i)
            for j, b in enumerate(blobs):
                d_field = math.hypot(b[0]-pred_x, b[1]-pred_y)
                d_img   = math.hypot(b[4]-pred_px_x, b[5]-pred_px_y)
                if i in self._underresolved_tracks and not self._blob_matches_track_merge(i, b):
                    continue
                if not self._is_plausible_match(i, b, d_field, d_img):
                    continue
                dist_scale = (self.MAX_REACQ_IN
                              if self._coast[i] > self.VISIBLE_COAST
                              else self.MAX_DIST_IN)
                dist_cost = d_field / max(dist_scale, 1.0)
                img_cost  = d_img / max(self._robot_max_px * 1.25, 1.0)
                qual_cost = b[6]
                real_cost[i, j] = dist_cost + 0.35 * img_cost + qual_cost

        if n == 0:
            return {}

        skip_cols = np.full((4, 4), INF, dtype=np.float64)
        for i in range(4):
            if self._pos[i] is not None:
                coast_frac = min(self._coast[i] / max(self.MAX_COAST, 1), 1.0)
                skip_cost_i = SKIP_COST * (1.0 + 0.35 * coast_frac)
                if self._coast[i] > self.VISIBLE_COAST:
                    skip_cost_i = SKIP_COST * (1.8 + 0.7 * coast_frac)
                if self._merge_recent[i] > 0:
                    skip_cost_i = min(skip_cost_i, SKIP_COST * 0.8)
            else:
                skip_cost_i = SKIP_COST * 0.9
            skip_cols[i, i] = skip_cost_i

        cost = np.hstack([real_cost, skip_cols])

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

    def _blob_matches_track_merge(self, track_id: int, blob) -> bool:
        pred_px_x, pred_px_y = self._predict_image_pos(track_id)
        contour = blob[8]
        if contour is None:
            return False
        dist = abs(self.cv2.pointPolygonTest(contour, (float(pred_px_x), float(pred_px_y)), True))
        return dist <= self._robot_max_px * 0.75

    def _predict_field_pos(self, track_id: int) -> Tuple[float, float]:
        if self._pos[track_id] is None:
            return (0.0, 0.0)
        px, py = self._pos[track_id]
        vx, vy = self._vel[track_id]
        coast = min(self._coast[track_id], self.MAX_COAST)
        horizon = min(1.0 + 0.15 * coast, 2.0)
        return (px + vx * horizon, py + vy * horizon)

    def _predict_image_pos(self, track_id: int) -> Tuple[float, float]:
        if self._pos_px[track_id] is None:
            if self._pos[track_id] is None:
                return (0.0, 0.0)
            pt = self.np.array([[[self._pos[track_id][0], self._pos[track_id][1]]]],
                               dtype=self.np.float32)
            ip = self.cv2.perspectiveTransform(pt, self._H_inv)[0][0]
            return (float(ip[0]), float(ip[1]))
        px, py = self._pos_px[track_id]
        vx, vy = self._vel_px[track_id]
        coast = min(self._coast[track_id], self.MAX_COAST)
        horizon = min(1.0 + 0.15 * coast, 2.0)
        return (px + vx * horizon, py + vy * horizon)

    def _update_track_motion(self, track_id: int, field_pos, image_pos) -> None:
        old_field = self._pos[track_id]
        old_image = self._pos_px[track_id]
        if old_field is not None:
            raw_vx = field_pos[0] - old_field[0]
            raw_vy = field_pos[1] - old_field[1]
            speed = math.hypot(raw_vx, raw_vy)
            if speed > self.MAX_SPEED_IN:
                scale = self.MAX_SPEED_IN / max(speed, 1e-9)
                raw_vx *= scale
                raw_vy *= scale
            prev_vx, prev_vy = self._vel[track_id]
            self._vel[track_id] = (
                prev_vx * 0.45 + raw_vx * 0.55,
                prev_vy * 0.45 + raw_vy * 0.55,
            )
        if old_image is not None:
            raw_vx_px = image_pos[0] - old_image[0]
            raw_vy_px = image_pos[1] - old_image[1]
            prev_vx_px, prev_vy_px = self._vel_px[track_id]
            self._vel_px[track_id] = (
                prev_vx_px * 0.45 + raw_vx_px * 0.55,
                prev_vy_px * 0.45 + raw_vy_px * 0.55,
            )

    def _motion_heading(self, track_id: int, fallback: float = 0.0) -> float:
        vx, vy = self._vel[track_id]
        if math.hypot(vx, vy) < 0.25:
            return fallback
        return math.atan2(vy, vx)

    def _is_plausible_match(self, track_id: int, blob, d_field: float, d_img: float) -> bool:
        contour_count = max(blob[7], 1)
        effective_hold = max(self._coast[track_id], self._merge_recent[track_id])
        if effective_hold > 0 and d_field > self.MAX_REACQ_IN:
            return False
        img_limit = self._robot_max_px * (
            1.3 + self.REACQ_PX_PAD * min(effective_hold, 4))
        if d_img > img_limit:
            return False
        if contour_count == 1 and blob[6] >= 0.9 and effective_hold > 0:
            return False
        if self._merge_recent[track_id] > 0 and contour_count == 1 and blob[6] >= 0.45:
            return False
        return True

    # ── merge group lifecycle ─────────────────────────────────────────────

    def _create_merge_group(self, track_ids: List[int], parent_id: int) -> MergeGroup:
        """
        Create a MergeGroup.  For 3+ tracks, record entry_order so we can
        detect permutations during the merge (fix f).
        """
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

        # Principal axis: direction of maximum spread among the entry positions
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
        ordered_positions = sorted(positions, key=lambda p: _proj(p[1], p[2]))
        entry_order = [p[0] for p in ordered_positions]

        # Seed peak_assignment with entry positions so we have valid state
        # immediately (guards against first-frame stale-peak scenario, fix g)
        peak_assignment = {tid: (px, py) for tid, px, py in positions}

        return MergeGroup(
            track_ids      = list(track_ids),
            entry_axis     = (ax, ay),
            parent_id      = parent_id,
            crossed        = False,
            entry_order    = entry_order,
            current_order  = list(entry_order),   # starts as identity permutation
            peak_assignment= peak_assignment,
        )

    def _update_crossing(self, mg: MergeGroup, contour) -> None:
        """
        Update per-frame merge state.

        2-robot merges: unchanged from v3.1 (crossed flag).
        3+ robot merges: update peak_assignment AND current_order so we can
        detect the full permutation at separation (fixes e, f, g).
        """
        cv2 = self.cv2
        peaks = self._split_contour(contour)
        n = len(mg.track_ids)

        # FIX (g): if we got fewer peaks than tracks, keep existing peak_assignment
        # rather than silently doing nothing.  This preserves the last good state.
        if len(peaks) < 2:
            # Fully overlapping — can't distinguish positions; preserve state.
            return

        M = cv2.moments(contour)
        if M["m00"] == 0:
            return
        mcx = M["m10"] / M["m00"]
        mcy = M["m01"] / M["m00"]
        ax, ay = mg.entry_axis

        def _proj(px, py): return (px - mcx) * ax + (py - mcy) * ay

        # ── 2-robot: original crossing logic (unchanged) ──────────────────
        if n == 2:
            p0, p1 = sorted(peaks[:2], key=lambda p: _proj(p[0], p[1]))
            dot = (p1[0] - p0[0]) * ax + (p1[1] - p0[1]) * ay
            mg.crossed = (dot < 0)
            return

        # ── 3+ robot: assign peaks to tracks then update current_order ────
        #
        # FIX (e) + (f): we assign each peak to the nearest track using the
        # CURRENT positions (which are themselves updated from peaks each frame
        # via the main update loop), not the frozen entry positions.  This
        # means prediction tracks the robots through the blob rather than
        # drifting back to entry.
        #
        # If peaks < n (fix g): assign what we can, leave the rest at their
        # last known positions.  current_order is only updated when we have
        # enough peaks to determine it.

        # Sort available peaks along entry axis
        sorted_peaks = sorted(peaks, key=lambda p: _proj(p[0], p[1]))

        # Build new assignment: for each track, find nearest unused peak
        new_assignment = dict(mg.peak_assignment)  # start from last good state
        used_peaks = set()

        # Use current positions (updated each frame from peaks) not entry positions
        track_current_px = []
        for tid in mg.track_ids:
            if self._pos_px[tid] is not None:
                track_current_px.append((tid, float(self._pos_px[tid][0]),
                                              float(self._pos_px[tid][1])))
            else:
                # Fall back to transformed field position
                if self._pos[tid] is not None:
                    pt = self.np.array([[[self._pos[tid][0], self._pos[tid][1]]]],
                                       dtype=self.np.float32)
                    ip = self.cv2.perspectiveTransform(pt, self._H_inv)[0][0]
                    track_current_px.append((tid, float(ip[0]), float(ip[1])))
                else:
                    track_current_px.append((tid, mcx, mcy))

        # Greedy nearest-peak assignment (globally optimal would need
        # Hungarian here too, but n<=4 peaks means greedy is fine with
        # the sorted-by-projection tie-breaking below)
        # Sort tracks by current projection so nearby tracks compete fairly
        track_current_px.sort(key=lambda t: _proj(t[1], t[2]))

        successfully_assigned = 0
        for tid, tx, ty in track_current_px:
            best_d, best_k = float('inf'), None
            for k, (px, py) in enumerate(sorted_peaks):
                if k in used_peaks:
                    continue
                d = math.hypot(px - tx, py - ty)
                if d < best_d:
                    best_d, best_k = d, k
            if best_k is not None:
                new_assignment[tid] = sorted_peaks[best_k]
                used_peaks.add(best_k)
                successfully_assigned += 1

        mg.peak_assignment = new_assignment

        # Update current_order: sort track IDs by their assigned peak projection
        # Only update if we assigned at least as many peaks as we have tracks
        # (if we had fewer peaks, the order might be partially stale — keep last)
        if successfully_assigned >= n:
            mg.current_order = sorted(
                mg.track_ids,
                key=lambda tid: _proj(*mg.peak_assignment[tid])
                if tid in mg.peak_assignment else 0.0
            )

    def _apply_separation(self, mg: MergeGroup) -> None:
        """
        Re-anchor track state on separation.

        2-robot merges: unchanged — swap if crossed.
        3+ robot merges: FIX (h) — apply the full permutation derived from
        entry_order vs current_order, then re-anchor each track to its
        current peak position.
        """
        for tid in mg.track_ids:
            self._merge_recent[tid] = self.MERGE_HOLD

        # ── 3+ robot case ─────────────────────────────────────────────────
        if len(mg.track_ids) != 2:
            self._apply_separation_multi(mg)
            return

        # ── 2-robot case (unchanged from v3.1) ────────────────────────────
        if mg.crossed:
            tids = mg.track_ids
            n    = len(tids)
            sp   = [self._pos[tids[k]]    for k in range(n)]
            sppx = [self._pos_px[tids[k]] for k in range(n)]
            sv   = [self._vel[tids[k]]    for k in range(n)]
            svpx = [self._vel_px[tids[k]] for k in range(n)]
            for k in range(n):
                self._pos[tids[k]]    = sp[n-1-k]
                self._pos_px[tids[k]] = sppx[n-1-k]
                self._vel[tids[k]]    = sv[n-1-k]
                self._vel_px[tids[k]] = svpx[n-1-k]
            print("[INFO] Separation WITH crossing — swapped: {}".format(
                sorted(mg.track_ids)))
        else:
            print("[INFO] Separation, no crossing — preserved: {}".format(
                sorted(mg.track_ids)))

    def _apply_separation_multi(self, mg: MergeGroup) -> None:
        """
        FIX (h): Apply the permutation mapping for 3+ robot merges.

        entry_order[k] is the track ID that was at position k along entry_axis
        when the merge started.  current_order[k] is the track ID whose peak
        is currently at position k along entry_axis.

        Interpretation: the robot now at slot k (current_order[k]) should
        inherit the identity of the robot that *entered* slot k (entry_order[k]).
        In other words, track entry_order[k] should get the state of current_order[k].

        Example:
          entry_order   = [0, 1, 2]   (left to right at merge start)
          current_order = [2, 0, 1]   (robots rearranged: R2 is now leftmost)
          → R0 should get R2's current peak (slot 0 belongs to entry R0)
          → R1 should get R0's current peak
          → R2 should get R1's current peak
        """
        entry   = mg.entry_order
        current = mg.current_order
        n       = len(entry)

        if len(current) != n:
            # current_order was never fully populated (e.g. peaks always < n)
            # Fall back to simple re-anchor without permutation
            print("[WARN] Multi-robot merge resolved with incomplete order info — "
                  "re-anchoring in place: {}".format(sorted(mg.track_ids)))
            self._reanchor_from_peaks(mg.track_ids, mg.peak_assignment)
            return

        # Build permutation: for each slot k, who's there now vs who should be
        # Slot k should have entry_order[k].  It currently has current_order[k].
        # So entry_order[k] ← state of current_order[k].

        # Snapshot current state for all tracks in the group
        snap_pos    = {tid: self._pos[tid]    for tid in mg.track_ids}
        snap_pos_px = {tid: self._pos_px[tid] for tid in mg.track_ids}
        snap_vel    = {tid: self._vel[tid]    for tid in mg.track_ids}
        snap_vel_px = {tid: self._vel_px[tid] for tid in mg.track_ids}

        swapped = []
        for k in range(n):
            dest_tid = entry[k]    # the ID that owns slot k
            src_tid  = current[k]  # who is physically at slot k right now
            if dest_tid == src_tid:
                continue
            # Assign src's current state to dest
            self._pos[dest_tid]    = snap_pos.get(src_tid)
            self._pos_px[dest_tid] = snap_pos_px.get(src_tid)
            self._vel[dest_tid]    = snap_vel.get(src_tid, (0.0, 0.0))
            self._vel_px[dest_tid] = snap_vel_px.get(src_tid, (0.0, 0.0))
            swapped.append((dest_tid, src_tid))

        # Now re-anchor each track to its peak for a clean separation position
        # (peak_assignment is keyed by track ID after the permutation above
        #  already updated the positions, so we re-anchor by dest_tid → peak
        #  of the physical robot now assigned to that ID)
        for k in range(n):
            dest_tid = entry[k]
            src_tid  = current[k]
            if src_tid in mg.peak_assignment:
                px, py = mg.peak_assignment[src_tid]
                pt = self.np.array([[[px, py]]], dtype=self.np.float32)
                fp = self.cv2.perspectiveTransform(pt, self._H_2d)[0][0]
                fx, fy = float(fp[0]), float(fp[1])
                if 0 <= fx <= 144 and 0 <= fy <= 144:
                    self._pos[dest_tid]    = (fx, fy)
                    self._pos_px[dest_tid] = (px, py)
                    self._vel[dest_tid]    = (0.0, 0.0)
                    self._vel_px[dest_tid] = (0.0, 0.0)

        if swapped:
            print("[INFO] Multi-robot merge resolved WITH permutation: "
                  "{} — swaps: {}".format(
                      sorted(mg.track_ids),
                      ", ".join("R{}←R{}".format(d, s) for d, s in swapped)))
        else:
            print("[INFO] Multi-robot merge resolved, no permutation: {}".format(
                sorted(mg.track_ids)))

    def _reanchor_from_peaks(self, track_ids, peak_assignment):
        """Fallback: re-anchor each track to its last-known peak, in-place."""
        reanchored = 0
        for tid in track_ids:
            if tid in peak_assignment:
                px, py = peak_assignment[tid]
                pt = self.np.array([[[px, py]]], dtype=self.np.float32)
                fp = self.cv2.perspectiveTransform(pt, self._H_2d)[0][0]
                fx, fy = float(fp[0]), float(fp[1])
                if 0 <= fx <= 144 and 0 <= fy <= 144:
                    self._pos[tid]    = (fx, fy)
                    self._pos_px[tid] = (px, py)
                    self._vel[tid]    = (0.0, 0.0)
                    self._vel_px[tid] = (0.0, 0.0)
                    reanchored += 1
        print("[INFO] Re-anchored {}/{} tracks (fallback)".format(
            reanchored, len(track_ids)))

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

        ball_mask = self._ball_color_mask(frame)
        if ball_mask is not None:
            fg = cv2.bitwise_and(fg, cv2.bitwise_not(ball_mask))

        return fg

    def _scaled_odd_kernel(self, frac: float, min_size: int):
        cv2 = self.cv2
        size = max(min_size, int(round(self._robot_max_px * frac)))
        if size % 2 == 0:
            size += 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def _ball_color_mask(self, frame):
        if not self.BALL_FILTER_ENABLED or self._field_mask is None:
            self._last_ball_mask = None
            return None

        cv2, np = self.cv2, self.np
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(
            hsv,
            np.array(self.BALL_GREEN_HSV_LO, dtype=np.uint8),
            np.array(self.BALL_GREEN_HSV_HI, dtype=np.uint8),
        )
        purple = cv2.inRange(
            hsv,
            np.array(self.BALL_PURPLE_HSV_LO, dtype=np.uint8),
            np.array(self.BALL_PURPLE_HSV_HI, dtype=np.uint8),
        )
        mask = cv2.bitwise_or(green, purple)
        mask = cv2.bitwise_and(mask, self._field_mask)

        open_kern = self._scaled_odd_kernel(self.BALL_MASK_OPEN_FRAC, 3)
        close_kern = self._scaled_odd_kernel(self.BALL_MASK_CLOSE_FRAC, 3)
        dilate_kern = self._scaled_odd_kernel(self.BALL_MASK_DILATE_FRAC, 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kern)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kern)
        mask = self._filter_contours_by_field_area(mask, self.BALL_MIN_FIELD_AREA_IN2)
        mask = cv2.dilate(mask, dilate_kern)
        self._last_ball_mask = mask
        return mask

    def _filter_contours_by_field_area(self, mask, min_area_in2: float):
        cv2, np = self.cv2, self.np
        if self._H_2d is None or min_area_in2 <= 0:
            return mask

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept = np.zeros_like(mask)
        for c in cnts:
            if len(c) < 3:
                continue
            if self._contour_field_area_in2(c) >= min_area_in2:
                cv2.drawContours(kept, [c], -1, 255, -1)
        return kept

    def _contour_field_area_in2(self, contour) -> float:
        cv2, np = self.cv2, self.np
        if self._H_2d is None:
            return 0.0
        approx = cv2.approxPolyDP(contour, 1.5, True)
        if len(approx) < 3:
            return 0.0
        pts = approx.reshape(-1, 2).astype(np.float32)
        field = cv2.perspectiveTransform(np.array([pts], dtype=np.float32),
                                         self._H_2d)[0]
        return abs(float(cv2.contourArea(field.reshape(-1, 1, 2))))

    # ── blob detector ─────────────────────────────────────────────────────

    def _foreground_contours(self, frame):
        cv2, np = self.cv2, self.np
        fg = self._foreground_mask(frame)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for c in cnts:
            if cv2.contourArea(c) < self.BLOB_MIN:
                continue
            if self._contour_field_area_in2(c) < self.BLOB_MIN_FIELD_AREA_IN2:
                continue
            valid.append(c)
        valid.sort(key=lambda c: -cv2.contourArea(c))
        all_split = []
        for c in valid:
            area = float(cv2.contourArea(c))
            subs = self._split_contour(c)
            per  = area / max(len(subs), 1)
            for sx, sy in subs:
                all_split.append((sx, sy, per))
        all_split.sort(key=lambda x: -x[2])
        return valid, [(int(sx), int(sy)) for sx, sy, _ in all_split]

    def _split_contour(self, contour):
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
        area = float(cv2.contourArea(contour))
        expected = self._expected_robot_count(area)
        if len(centers) < expected:
            retry = self._split_contour_relaxed(mask, dist, expected)
            if len(retry) > len(centers):
                centers = retry
        return centers

    def _expected_robot_count(self, contour_area: float) -> int:
        robot_area = max(self._robot_max_px * self._robot_max_px, 1.0)
        ratio = contour_area / robot_area
        if ratio >= self.EXPECT_4WAY_AREA:
            return 4
        if ratio >= self.EXPECT_3WAY_AREA:
            return 3
        if ratio >= self.EXPECT_2WAY_AREA:
            return 2
        return 1

    def _split_contour_relaxed(self, mask, dist, expected: int):
        cv2, np = self.cv2, self.np
        if dist.max() == 0:
            return []
        peak_ratio = (self.RELAXED_PEAK_RATIO_MULTI
                      if expected >= 3 else self.RELAXED_PEAK_RATIO_PAIR)
        _, peak_mask = cv2.threshold(dist, dist.max() * peak_ratio, 255, cv2.THRESH_BINARY)
        peak_mask = peak_mask.astype(np.uint8)
        sep_size = self.RELAXED_SEP_MULTI if expected >= 3 else self.RELAXED_SEP_PAIR
        k_sep = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sep_size, sep_size))
        peak_mask = cv2.erode(peak_mask, k_sep)
        peak_mask = cv2.bitwise_and(peak_mask, mask)
        n_labels, labels = cv2.connectedComponents(peak_mask)
        centers = []
        for lbl in range(1, n_labels):
            comp = (labels == lbl).astype(np.uint8)
            M = cv2.moments(comp)
            if M["m00"] > 0:
                centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        return centers

    def _find_underresolved_tracks(self, blobs: list):
        blocked = set()
        seen = {}
        for blob in blobs:
            pid = blob[3]
            if pid in seen:
                continue
            seen[pid] = (blob[8], blob[7])

        for contour, split_count in seen.values():
            if contour is None:
                continue
            expected = self._expected_robot_count(float(self.cv2.contourArea(contour)))
            nearby_tracks = []
            for track_id in range(4):
                if self._pos[track_id] is None and self._pos_px[track_id] is None:
                    continue
                pred_px_x, pred_px_y = self._predict_image_pos(track_id)
                dist = self.cv2.pointPolygonTest(
                    contour, (float(pred_px_x), float(pred_px_y)), True)
                if dist >= -self._robot_max_px * 0.75:
                    nearby_tracks.append(track_id)
            expected = max(expected, len(nearby_tracks))
            if expected <= split_count:
                continue
            blocked.update(nearby_tracks)
        return blocked

    def _get_blobs(self, frame):
        cv2, np = self.cv2, self.np
        fg = self._foreground_mask(frame)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for parent_id, c in enumerate(cnts):
            area = float(cv2.contourArea(c))
            if area < self.BLOB_MIN:
                continue
            field_area = self._contour_field_area_in2(c)
            if field_area < self.BLOB_MIN_FIELD_AREA_IN2:
                continue
            _bm = np.zeros(self._field_mask.shape, dtype=np.uint8)
            cv2.drawContours(_bm, [c], -1, 255, -1)
            _dr = cv2.distanceTransform(_bm, cv2.DIST_L2, 5)
            if _dr.max() < self.MIN_RADIUS_PX:
                continue
            sub_centers = self._split_contour(c)
            sub_centers = self._augment_subcenters_with_track_priors(c, sub_centers)
            if not sub_centers:
                continue
            per_area = area / len(sub_centers)
            quality_penalty = self._blob_quality_penalty(area, per_area, len(sub_centers))
            for cx, cy in sub_centers:
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                fp = cv2.perspectiveTransform(pt, self._H_2d)[0][0]
                fx, fy = float(fp[0]), float(fp[1])
                if not (0 <= fx <= 144 and 0 <= fy <= 144):
                    continue
                candidates.append((fx, fy, 0.0, per_area, parent_id,
                                    int(cx), int(cy), quality_penalty,
                                    len(sub_centers), c))

        candidates.sort(key=lambda b: -b[3])
        return [(fx, fy, hdg, pid, cx, cy, qual, nsplit, cnt)
                for fx, fy, hdg, _a, pid, cx, cy, qual, nsplit, cnt in candidates]

    def _augment_subcenters_with_track_priors(self, contour, centers):
        expected = self._expected_robot_count(float(self.cv2.contourArea(contour)))
        if len(centers) >= expected:
            return centers

        nearby = []
        for track_id in range(4):
            if self._pos[track_id] is None and self._pos_px[track_id] is None:
                continue
            px, py = self._predict_image_pos(track_id)
            dist = self.cv2.pointPolygonTest(contour, (float(px), float(py)), True)
            if dist >= -self._robot_max_px * self.MERGE_PRIOR_PAD_PX:
                nearby.append((dist, float(px), float(py)))

        target = min(max(expected, len(nearby)), 4)
        if len(centers) >= target or not nearby:
            return centers

        augmented = list(centers)
        min_sep = max(self._robot_max_px * self.MERGE_PRIOR_MIN_SEP, 6.0)
        nearby.sort(reverse=True)
        for _dist, px, py in nearby:
            too_close = any(math.hypot(px-cx, py-cy) < min_sep for cx, cy in augmented)
            if too_close:
                continue
            augmented.append((px, py))
            if len(augmented) >= target:
                break
        return augmented

    def _blob_quality_penalty(self, area: float, per_area: float, split_count: int) -> float:
        robot_area = max(self._robot_max_px * self._robot_max_px, 1.0)
        ratio = per_area / robot_area
        penalty = min(abs(math.log(max(ratio, 1e-6))), 2.5) * 0.28
        if split_count == 1 and area > robot_area * 1.8:
            penalty += min((area / robot_area) - 1.8, 2.0) * 0.7
        return penalty


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
                    np.array([[0,0],[144,0],[144,144],[0,144]], np.float32))
                return ordered, H
        return None


def _project_field_points(cv2, np, H_inv, points_in):
    pts = np.array([points_in], dtype=np.float32)
    img = cv2.perspectiveTransform(pts, H_inv)[0]
    return [(float(p[0]), float(p[1])) for p in img]


def _project_image_points(cv2, np, H_2d, points_px):
    pts = np.array([points_px], dtype=np.float32)
    field = cv2.perspectiveTransform(pts, H_2d)[0]
    return [(float(p[0]), float(p[1])) for p in field]


def _detect_robot_edge_profile(cv2, np, frame, fg_mask, tracker, center_px):
    """
    Find the robot's floor contact point from the local foreground silhouette.

    Tracker poses are blob centers, not ground-contact centers. For the cube
    base we choose the connected foreground component nearest the tracked point,
    then use Canny edges on its lower silhouette to estimate where the robot
    touches the field.
    """
    h, w = frame.shape[:2]
    cx, cy = center_px
    pad = max(22, int(tracker._robot_max_px * 1.05))
    x0 = max(0, int(cx - pad)); x1 = min(w, int(cx + pad))
    y0 = max(0, int(cy - pad * 1.5)); y1 = min(h, int(cy + pad * 1.2))
    fallback_h = tracker._robot_max_px * 0.85
    fallback_box = (
        cx - tracker._robot_max_px * 0.35,
        cy - fallback_h,
        cx + tracker._robot_max_px * 0.35,
        cy,
    )
    if x1 <= x0 or y1 <= y0:
        return {
            "base_px": center_px,
            "top_y": cy - fallback_h,
            "bbox": fallback_box,
        }

    roi = frame[y0:y1, x0:x1]
    fg_roi = fg_mask[y0:y1, x0:x1]
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg_roi, 8)
    if n_labels <= 1:
        return {
            "base_px": center_px,
            "top_y": cy - fallback_h,
            "bbox": fallback_box,
        }

    local_cx = float(cx - x0)
    local_cy = float(cy - y0)
    center_label = labels[int(np.clip(round(local_cy), 0, labels.shape[0]-1)),
                          int(np.clip(round(local_cx), 0, labels.shape[1]-1))]
    best_lbl = None
    best_score = -1.0
    for lbl in range(1, n_labels):
        area = float(stats[lbl, cv2.CC_STAT_AREA])
        if area < max(30.0, tracker.BLOB_MIN * 0.12):
            continue
        ccx, ccy = centroids[lbl]
        dist = math.hypot(float(ccx) - local_cx, float(ccy) - local_cy)
        score = area / (1.0 + dist / max(tracker._robot_max_px, 1.0))
        if center_label == lbl:
            score *= 2.0
        if score > best_score:
            best_score = score
            best_lbl = lbl

    if best_lbl is None:
        return {
            "base_px": center_px,
            "top_y": cy - fallback_h,
            "bbox": fallback_box,
        }

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 135)
    comp = (labels == best_lbl).astype(np.uint8) * 255
    edges = cv2.bitwise_and(edges, comp)

    comp_ys, comp_xs = np.where(comp > 0)
    if len(comp_xs) < 8:
        return {
            "base_px": center_px,
            "top_y": cy - fallback_h,
            "bbox": fallback_box,
        }

    ys, xs = np.where(edges > 0)
    if len(xs) >= 8:
        edge_rx = max(8.0, tracker._robot_max_px * 0.62)
        edge_ry = max(10.0, tracker._robot_max_px * 1.05)
        edge_keep = (
            ((xs.astype(np.float32) - local_cx) / edge_rx) ** 2 +
            ((ys.astype(np.float32) - local_cy) / edge_ry) ** 2
        ) <= 1.0
        if int(edge_keep.sum()) >= 8:
            xs = xs[edge_keep]
            ys = ys[edge_keep]

    comp_rx = max(8.0, tracker._robot_max_px * 0.65)
    comp_ry = max(10.0, tracker._robot_max_px * 1.10)
    comp_keep = (
        ((comp_xs.astype(np.float32) - local_cx) / comp_rx) ** 2 +
        ((comp_ys.astype(np.float32) - local_cy) / comp_ry) ** 2
    ) <= 1.0
    if int(comp_keep.sum()) >= 8:
        comp_xs = comp_xs[comp_keep]
        comp_ys = comp_ys[comp_keep]

    comp_h = max(1.0, float(comp_ys.max() - comp_ys.min() + 1))
    lower_y = comp_ys.min() + comp_h * 0.58

    lower_edges = ys >= lower_y
    if int(lower_edges.sum()) >= 5:
        edge_ys = ys[lower_edges]
        edge_xs = xs[lower_edges]
        base_y_local = float(np.percentile(edge_ys, 92))
        band = edge_ys >= base_y_local - max(3.0, comp_h * 0.16)
        base_x_local = float(np.median(edge_xs[band] if int(band.sum()) else edge_xs))
    else:
        base_y_local = float(np.percentile(comp_ys, 96))
        band = comp_ys >= base_y_local - max(3.0, comp_h * 0.14)
        base_x_local = float(np.median(comp_xs[band] if int(band.sum()) else comp_xs))

    comp_box = (
        x0 + float(np.percentile(comp_xs, 4)),
        y0 + float(np.percentile(comp_ys, 4)),
        x0 + float(np.percentile(comp_xs, 96)),
        y0 + float(np.percentile(comp_ys, 96)),
    )

    if len(xs) >= 8:
        canny_box = (
            x0 + float(np.percentile(xs, 3)),
            y0 + float(np.percentile(ys, 3)),
            x0 + float(np.percentile(xs, 97)),
            y0 + float(np.percentile(ys, 97)),
        )
        edge_box = (
            min(canny_box[0], comp_box[0]),
            min(canny_box[1], comp_box[1]),
            max(canny_box[2], comp_box[2]),
            max(canny_box[3], comp_box[3]),
        )
        top_y_local = float(np.percentile(ys, 8))
    else:
        edge_box = comp_box
        top_y_local = float(np.percentile(comp_ys, 5))

    return {
        "base_px": (x0 + base_x_local, y0 + base_y_local),
        "top_y": y0 + top_y_local,
        "bbox": edge_box,
    }


def _projected_bottom_for_side(cv2, np, H_inv, field_center, side_in):
    half = side_in / 2.0
    bottom_field = [
        (field_center[0] - half, field_center[1] - half),
        (field_center[0] + half, field_center[1] - half),
        (field_center[0] + half, field_center[1] + half),
        (field_center[0] - half, field_center[1] + half),
    ]
    return _project_field_points(cv2, np, H_inv, bottom_field)


def _choose_cube_side_in(cv2, np, H_inv, field_center, edge_bbox, tracker):
    x_min, _y_min, x_max, _y_max = edge_bbox
    target_min = x_min - 2.0
    target_max = x_max + 2.0
    lo = max(8.0, tracker.ROBOT_SIZE_IN * 0.45)
    hi = tracker.ROBOT_SIZE_IN * 1.20
    best = hi
    for _ in range(10):
        mid = (lo + hi) * 0.5
        bottom = _projected_bottom_for_side(cv2, np, H_inv, field_center, mid)
        bx_min = min(p[0] for p in bottom)
        bx_max = max(p[0] for p in bottom)
        if bx_min <= target_min and bx_max >= target_max:
            best = mid
            hi = mid
        else:
            lo = mid
    return best


def _draw_wire_cube(cv2, np, dbg, frame, fg_mask, tracker, H_inv,
                    pose, center_px, color):
    profile = _detect_robot_edge_profile(
        cv2, np, frame, fg_mask, tracker, center_px)
    base_px = profile["base_px"]
    top_y = profile["top_y"]
    edge_bbox = profile["bbox"]
    if tracker._H_2d is not None:
        field_center = _project_image_points(cv2, np, tracker._H_2d, [base_px])[0]
    else:
        field_center = (pose.x_in, pose.y_in)

    side_in = _choose_cube_side_in(cv2, np, H_inv, field_center, edge_bbox, tracker)
    bottom = _projected_bottom_for_side(cv2, np, H_inv, field_center, side_in)
    h, w = dbg.shape[:2]
    if not all(-w <= x <= 2*w and -h <= y <= 2*h for x, y in bottom):
        return

    height_px = base_px[1] - top_y
    height_px = float(np.clip(height_px,
                              tracker._robot_max_px * 0.45,
                              tracker._robot_max_px * 1.45))
    top = [(x, y - height_px) for x, y in bottom]

    bottom_i = [(int(round(x)), int(round(y))) for x, y in bottom]
    top_i = [(int(round(x)), int(round(y))) for x, y in top]

    for pts, thickness in ((bottom_i, 1), (top_i, 2)):
        for j in range(4):
            cv2.line(dbg, pts[j], pts[(j+1) % 4], color, thickness, cv2.LINE_AA)
    for j in range(4):
        cv2.line(dbg, bottom_i[j], top_i[j], color, 2, cv2.LINE_AA)


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
    debug_enable_hitboxes = False,
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
    if PROCESS_EVERY_SOURCE_FRAME:
        frame_step = 1
    print("[INFO] Processing every {} frames (~{:.1f} fps output)".format(
        frame_step, video_fps / frame_step))
    if debug:
        print("[INFO] Saving debug frame about every {} source frames".format(debug_every_n))
        if debug_enable_hitboxes:
            print("[INFO] Debug robot wireframe hitboxes enabled.")

    csv_path    = os.path.join(output_dir, "robot_positions.csv")
    wpilog_path = os.path.join(output_dir, "match_log.wpilog")
    debug_dir   = os.path.join(output_dir, "tracker_debug") if debug else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        for name in os.listdir(debug_dir):
            if name.endswith(".jpg"):
                try:
                    os.remove(os.path.join(debug_dir, name))
                except OSError:
                    pass

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
            ordered, np.array([[0,0],[144,0],[144,144],[0,144]], np.float32))
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
    merge_debug_hold = 0
    dense_merge_hold = 0
    next_debug_source_frame = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_num = frame_num
        match_time_s = current_frame_num / video_fps - start_offset_sec
        timestamp_us = max(0, int(match_time_s * 1_000_000))
        frame_num   += 1

        process_dense = dense_merge_hold > 0 or bool(tracker._merge_groups)
        if not process_dense and (current_frame_num - start_frame) % frame_step != 0:
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
        merge_active = bool(tracker._merge_groups)
        recent_merge = any(v > 0 for v in tracker._merge_recent)
        if merge_active or recent_merge:
            dense_merge_hold = tracker.MERGE_DEBUG_HOLD
        elif dense_merge_hold > 0:
            dense_merge_hold -= 1
        if merge_active:
            merge_debug_hold = tracker.MERGE_DEBUG_HOLD
        elif merge_debug_hold > 0:
            merge_debug_hold -= 1

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
            fg_debug = None
            if ordered is not None:
                cv2.polylines(dbg, [ordered.astype(np.int32).reshape(-1,1,2)],
                              True, (0, 200, 0), 2)

            if tracker._bg is not None:
                fg_debug = tracker._foreground_mask(frame)
                all_cnts, split_centers = tracker._foreground_contours(frame)
                if tracker._last_ball_mask is not None:
                    ball_cnts, _ = cv2.findContours(
                        tracker._last_ball_mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    for bc in ball_cnts:
                        if cv2.contourArea(bc) >= 20:
                            cv2.drawContours(dbg, [bc], -1, (255, 0, 255), 1)
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

            for mg in tracker._merge_groups.values():
                pos_list = [tracker._pos_px[t] for t in mg.track_ids
                            if tracker._pos_px[t] is not None]
                if pos_list:
                    mcx = int(sum(p[0] for p in pos_list) / len(pos_list))
                    mcy = int(sum(p[1] for p in pos_list) / len(pos_list))
                    ax, ay = mg.entry_axis
                    ex = int(mcx+ax*50); ey = int(mcy+ay*50)
                    # Show permutation state in debug overlay
                    if len(mg.track_ids) == 2:
                        col = (0, 80, 255) if mg.crossed else (0, 220, 100)
                        lbl2 = "+".join("R{}".format(t) for t in mg.track_ids)
                        lbl2 += " CROSSED" if mg.crossed else " ok"
                    else:
                        col = (255, 160, 0)
                        entry_s   = "".join(str(t) for t in mg.entry_order)
                        current_s = "".join(str(t) for t in mg.current_order)
                        lbl2 = "+".join("R{}".format(t) for t in mg.track_ids)
                        lbl2 += " [{}→{}]".format(entry_s, current_s)
                    cv2.arrowedLine(dbg, (mcx, mcy), (ex, ey), col, 2)
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
                if debug_enable_hitboxes and fg_debug is not None:
                    _draw_wire_cube(cv2, np, dbg, frame, fg_debug,
                                    tracker, H_inv, p, (ix, iy), col)
                cv2.putText(dbg, "R{}".format(i), (ix-10, iy-24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                vx_px, vy_px = tracker._vel_px[i]
                speed_px = math.hypot(vx_px, vy_px)
                if speed_px >= 0.35:
                    scale = 28.0 / speed_px
                    dx = int(round(vx_px * scale))
                    dy = int(round(vy_px * scale))
                    cv2.arrowedLine(dbg, (ix, iy), (ix+dx, iy+dy),
                                    (255,255,255), 2)
                n_drawn += 1

            n_merged = sum(len(mg.track_ids) for mg in tracker._merge_groups.values())
            lbl = ("init" if not tracker._initialized else
                   "{}/4 ({} merged)".format(n_drawn, n_merged) if n_merged else
                   "{}/4".format(n_drawn))
            cv2.putText(dbg, "t={:.2f}s f={}  [{}]".format(
                        match_time_s, current_frame_num, lbl),
                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            base_debug_interval = max(1, debug_every_n)
            save_debug = (current_frame_num >= next_debug_source_frame
                          or (SAVE_ALL_DEBUG_AROUND_MERGES and
                              (merge_active or merge_debug_hold > 0)))
            if save_debug:
                cv2.imwrite(os.path.join(debug_dir,
                            "frame_{:06d}.jpg".format(current_frame_num)), dbg)
                if current_frame_num >= next_debug_source_frame:
                    while current_frame_num >= next_debug_source_frame:
                        next_debug_source_frame += base_debug_interval

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
|  Debug overlay: 2-robot merges show CROSSED/ok; 3+ show          |
|  [entry_order→current_order] permutation string.                 |
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
    p.add_argument("--debug-enable-hitboxes", action="store_true",
                   help="Draw robot wireframe hitboxes in debug frames")
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
        debug_enable_hitboxes = args.debug_enable_hitboxes,
        manual_corners_px = corners,
    )


if __name__ == "__main__":
    main()