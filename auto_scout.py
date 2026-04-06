# -*- coding: utf-8 -*-
"""
auto_scout.py — FTC DECODE robot tracker  (v3.1)

Base: v3 topology-based merge tracking (the "better but still buggy" version).
Changes in v3.1 — targeted fixes only, zero structural changes:

  a) NECK-BREAKING  — after the standard morphology pass, erode the fg mask by
     a physics-derived kernel (~15 % of robot pixel width) then dilate back.
     Severs thin noise bridges between two nearby robots before contour-finding,
     preventing spurious merges entirely.  Kernel size is computed from the
     homography + known 18" robot footprint, so it auto-scales to any camera.

  b) UNCAPPED CANDIDATE POOL  — _get_blobs now returns every valid sub-centre,
     not capped at 4.  Previously a merged blob produced 2 split-centres and
     consumed 2 of the 4 slots, starving the other 2 solo robots.

  c) GLOBAL OPTIMAL ASSIGNMENT  — replaced the greedy nearest-neighbour loop
     with scipy.optimize.linear_sum_assignment (Hungarian algorithm).  Greedy
     processed tracks 0→3 in order, so track 0 always grabbed the nearest blob
     first; downstream tracks got leftovers.  A greedy fallback is used when
     scipy is not installed.

  d) SOFT DISTANCE COST  — distance is now a continuous cost (d / MAX_DIST_IN),
     not a hard gate.  Robots that sprint beyond 30" between sampled frames are
     no longer dropped.

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
    """
    State for 2 (or more) tracks sharing a single foreground blob.

    track_ids   : tracks sorted by projection onto entry_axis at merge time.
                  Index 0 = most "negative" end of axis.
    entry_axis  : unit vector (ax, ay) in IMAGE pixel space pointing from
                  track_ids[0] toward track_ids[-1] at merge time.
    parent_id   : contour parent_id of the current merged blob.
    crossed     : True if dist-transform peaks have swapped sides relative to
                  entry ordering.  Updated every frame during the merge.
    """
    track_ids  : List[int]             = dc_field(default_factory=list)
    entry_axis : Tuple[float, float]   = (1.0, 0.0)
    parent_id  : int                   = -1
    crossed    : bool                  = False


# ─────────────────────────────────────────────────────────────────────────────
# RobotTracker
# ─────────────────────────────────────────────────────────────────────────────
class RobotTracker:
    """
    Detect and track 4 robots using median background subtraction plus
    topology-aware merge handling.

    Normal frames  : globally optimal assignment of blobs to tracks.
    Merged frames  : dist-transform peaks watched inside the merged blob;
                     crossing detected by monitoring which side of the entry
                     axis each peak is on.
    Separation     : if crossed → swap IDs; else → preserve IDs.
    """

    N_BG_SAMPLES  = 80
    FG_THRESH     = 25
    BLOB_MIN      = 300
    MIN_RADIUS_PX = 10
    KERNEL_PX     = 9
    MAX_COAST     = 60
    MAX_DIST_IN   = 30.0   # scale for soft distance cost
    ROBOT_SIZE_IN = 18.0   # FTC max robot footprint (inches)

    def __init__(self, cv2, np):
        self.cv2, self.np = cv2, np
        self._bg           = None
        self._field_mask   = None
        self._H_2d         = None
        self._H_inv        = None
        self._kern         = cv2.getStructuringElement(
                                 cv2.MORPH_ELLIPSE, (self.KERNEL_PX, self.KERNEL_PX))
        self._neck_kern    = None   # set in setup()
        self._robot_max_px = 60     # safe default; overwritten in setup()

        self.tracked_poses = [RobotPose() for _ in range(4)]
        self._pos          = [None] * 4   # (x_in, y_in) field coords
        self._pos_px       = [None] * 4   # (cx_px, cy_px) image coords
        self._coast        = [999]  * 4
        self._initialized  = False

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

        # ── robot pixel footprint from homography ─────────────────────────
        # Project ROBOT_SIZE_IN inches along both axes from the field centre;
        # take the larger pixel distance.  Auto-scales to any camera setup.
        c_f = np.array([[[72.0, 72.0]]], dtype=np.float32)
        r_f = np.array([[[72.0 + self.ROBOT_SIZE_IN, 72.0]]], dtype=np.float32)
        u_f = np.array([[[72.0, 72.0 + self.ROBOT_SIZE_IN]]], dtype=np.float32)
        c_px = cv2.perspectiveTransform(c_f, self._H_inv)[0][0]
        r_px = cv2.perspectiveTransform(r_f, self._H_inv)[0][0]
        u_px = cv2.perspectiveTransform(u_f, self._H_inv)[0][0]
        px_x = math.hypot(r_px[0]-c_px[0], r_px[1]-c_px[1])
        px_y = math.hypot(u_px[0]-c_px[0], u_px[1]-c_px[1])
        self._robot_max_px = max(px_x, px_y)
        print("[INFO] Robot pixel footprint: {:.1f} px / 18 in".format(
            self._robot_max_px))

        # Neck-breaking kernel: ~15 % of robot width severs bridges
        # narrower than ~2.7 " without eroding real robot blobs.
        neck_r = max(3, int(self._robot_max_px * 0.15))
        self._neck_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (neck_r*2+1, neck_r*2+1))
        print("[INFO] Neck-breaking kernel radius: {} px".format(neck_r))

        # ── median background ─────────────────────────────────────────────
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

        blobs = self._get_blobs(frame)

        # ── initialise on first frame with ≥4 blobs ──────────────────────
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
            self._initialized = True
            return self.tracked_poses

        # ── globally optimal assignment ───────────────────────────────────
        assignment = self._assign(blobs)

        # ── detect merged blobs ───────────────────────────────────────────
        pid_to_contour = {}
        for b in blobs:
            if b[3] not in pid_to_contour:
                pid_to_contour[b[3]] = b[6]

        pid_to_tracks = defaultdict(list)
        for ti, bi in assignment.items():
            pid_to_tracks[blobs[bi][3]].append(ti)
        merged_pids = {pid: tl for pid, tl in pid_to_tracks.items() if len(tl) > 1}

        # ── update / create merge groups ──────────────────────────────────
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

        # ── resolve separations ───────────────────────────────────────────
        for key in list(self._merge_groups.keys()):
            if key not in active_keys:
                mg = self._merge_groups.pop(key)
                self._apply_separation(mg)

        # ── tracks currently inside a merge (position frozen) ────────────
        merged_tracks = set()
        for mg in self._merge_groups.values():
            merged_tracks.update(mg.track_ids)

        # ── apply per-track updates ───────────────────────────────────────
        for i in range(4):
            if i in assignment:
                fx, fy, hdg, pid, cx, cy, _cnt = blobs[assignment[i]]
                self._coast[i] = 0
                self.tracked_poses[i].x_in    = fx
                self.tracked_poses[i].y_in    = fy
                self.tracked_poses[i].heading = hdg
                self.tracked_poses[i].visible = True
                if i not in merged_tracks:
                    self._pos[i]    = (fx, fy)
                    self._pos_px[i] = (cx, cy)
            else:
                # Cap coast counter so it never overflows and the track
                # remains re-acquirable if a blob reappears near it later.
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
        Build a 4×(N+4) cost matrix where the last 4 columns are "skip" slots —
        one per track — at a fixed skip cost.

        Adding skip columns means the solver can leave a track unassigned (by
        routing it to its own skip slot) rather than forcing it onto a distant
        blob it doesn't actually own.  This is the critical fix for coasting
        tracks stealing blobs from their real owners.

        Skip cost = 2.0 (twice the scale distance).  A track only skips when
        every real blob is more than 2× MAX_DIST_IN away — i.e. truly out of
        reach.  Coasting tracks with no prior use skip cost 1.0 so they
        preferentially stay unassigned until they have a position to anchor on.

        Cost for real blobs = distance_inches / MAX_DIST_IN (soft, no cutoff).
        """
        np = self.np
        n = len(blobs)

        # Build real-blob cost columns
        INF       = 1e9
        SKIP_COST = 2.0   # skip is always available; only chosen if all blobs are far

        real_cost = np.full((4, max(n, 1)), INF, dtype=np.float64)
        for i in range(4):
            if self._pos[i] is None:
                # No prior position yet — neutral cost so we don't steal blobs
                real_cost[i, :] = SKIP_COST * 0.9
                continue
            px, py = self._pos[i]
            for j, b in enumerate(blobs):
                d = math.hypot(b[0]-px, b[1]-py)
                real_cost[i, j] = d / self.MAX_DIST_IN

        if n == 0:
            return {}

        # Append one skip column per track (diagonal identity block)
        skip_cols = np.full((4, 4), INF, dtype=np.float64)
        for i in range(4):
            if self._pos[i] is not None:
                # Long-coasting tracks get a lower skip cost so the solver
                # prefers assigning them to a nearby blob over skipping again.
                # coast=0 → skip_cost=SKIP_COST (normal)
                # coast≥MAX_COAST → skip_cost=0.5 (eagerly re-acquire)
                coast_frac = min(self._coast[i] / max(self.MAX_COAST, 1), 1.0)
                skip_cost_i = SKIP_COST * (1.0 - 0.75 * coast_frac)
            else:
                skip_cost_i = SKIP_COST * 0.9
            skip_cols[i, i] = skip_cost_i

        cost = np.hstack([real_cost, skip_cols])   # shape (4, n+4)

        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
            assignment = {}
            for r, c in zip(row_ind, col_ind):
                if c < n:   # skip columns live at index n..n+3
                    assignment[r] = c
        except ImportError:
            # Greedy fallback
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
            return   # fully overlapping — preserve current state
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
        if mg.crossed:
            tids = mg.track_ids
            n    = len(tids)
            sp   = [self._pos[tids[k]]    for k in range(n)]
            sppx = [self._pos_px[tids[k]] for k in range(n)]
            for k in range(n):
                self._pos[tids[k]]    = sp[n-1-k]
                self._pos_px[tids[k]] = sppx[n-1-k]
            print("[INFO] Separation WITH crossing — swapped: {}".format(
                sorted(mg.track_ids)))
        else:
            print("[INFO] Separation, no crossing — preserved: {}".format(
                sorted(mg.track_ids)))

    # ── foreground mask ───────────────────────────────────────────────────

    def _foreground_mask(self, frame):
        """
        Background-subtraction fg mask with neck-breaking.

        After the standard morphology pass, erode then dilate by the
        physics-derived neck kernel.  This severs bridges narrower than
        ~15 % of robot width without shrinking the robot blobs themselves.
        """
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

    # ── blob detector ─────────────────────────────────────────────────────

    def _foreground_contours(self, frame):
        """Used by the debug overlay only."""
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

    def _split_contour(self, contour):
        """Distance-transform peak detection; one centre per robot."""
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
        return centers

    def _get_blobs(self, frame):
        """
        Return ALL valid sub-centres from ALL large-enough contours as
        (fx_in, fy_in, heading, parent_id, cx_px, cy_px, contour).

        Not capped at 4 — the assignment step picks the best 4.
        Previously the cap caused a merged blob (2 sub-centres) to starve
        the other 2 solo robots out of the candidate pool.
        """
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
            sub_centers = self._split_contour(c)
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

            # 18" reference circle
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
                cv2.putText(dbg, "R{}".format(i), (ix-10, iy-24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
|  Debug frames show merge axis + CROSSED/ok, and 18" ref circle.  |
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