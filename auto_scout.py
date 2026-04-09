# -*- coding: utf-8 -*-
"""
auto_scout.py — FTC DECODE robot tracker  (v4.0)

Complete architectural rewrite.  The v3.x approach (background subtraction +
hand-crafted HSV histograms) was fundamentally unreliable for robots with
similar colouring and suffered from no motion model, making fast-moving robots
steal each other's detections every frame.

v4.0 architecture
─────────────────
  DETECT   Background subtraction (kept) but blobs are now just candidate
           detections, not the primary identity source.

  PREDICT  Each track maintains a Kalman filter [x, y, vx, vy] → [x, y].
           Every frame the filter predicts where each robot *should* be
           before any blobs are seen.  Robots that sprint between frames are
           handled correctly because we predict their trajectory, not just
           their last pixel position.

  ASSOCIATE Hungarian (scipy) on a joint cost matrix:
              cost = α·mahal_dist  +  β·appearance_dist  +  γ·size_cost
                   + hard color_penalty when locked classes differ
           Mahalanobis distance from Kalman prediction replaces naive Euclidean.
           appearance_dist uses deep MobileNetV3 crop embeddings when PyTorch
           is available, falling back to full H×S×V histograms.

  RE-ID    Appearance bank: per-track EMA of deep feature vectors (or HSV
           histograms).  Updated only on high-confidence assignments.
           Color class locked after 15 consistent observations.

  MERGE    Simplified: if a blob is >1.6× expected area AND contains ≥2 Kalman
           predicted positions, run distance-transform splitting / watershed.
           No complex topology state machine needed — Kalman predictions make
           split-point disambiguation trivial.

Usage:
  python3 auto_scout.py --no-download --video-path match.mp4 \\
      --corners field_corners.json [--debug] [--start-offset 1.0]

Hard dependencies:
  pip install opencv-python numpy progress

Soft dependencies (dramatically improve quality):
  pip install scipy          # optimal assignment (strongly recommended)
  pip install torch torchvision  # deep ReID embeddings (strongly recommended)
"""

import argparse
import csv
import json
import math
import os
import struct
import sys
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Tuple


def _require(package, pip_name=None):
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        name = pip_name or package
        print("[ERROR] Missing: {}\n  Install: pip install {}".format(name, name))
        sys.exit(1)


def _try_import(package):
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        return None


def _make_bar(label, max_val):
    try:
        from progress.bar import Bar
        return Bar(label, max=max_val,
                   suffix="%(percent).0f%% %(elapsed_td)s ETA %(eta_td)s")
    except ImportError:
        class _FB:
            def __init__(self, lbl, total):
                self._lbl = lbl; self._total = max(total, 1); self._n = 0
                print("[{}] 0%".format(lbl), end="", flush=True)
            def next(self):
                self._n += 1
                pct = int(self._n / self._total * 100)
                if self._n % max(1, self._total // 20) == 0 or self._n == self._total:
                    print("\r[{}] {}%".format(self._lbl, pct), end="", flush=True)
            def finish(self): print()
        return _FB(label, max_val)


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
            if n > max_bytes: break
            if v < (1 << (8 * n)):
                return v.to_bytes(n, "little"), n
        return v.to_bytes(max_bytes, "little"), max_bytes

    def _write_record(self, eid, ts, data):
        eb, el = self._encode_int(eid, 4)
        sb, sl = self._encode_int(len(data), 4)
        tb, tl = self._encode_int(ts, 8)
        bf = ((el-1)&3) | (((sl-1)&3)<<2) | (((tl-1)&7)<<4)
        self._fh.write(struct.pack("<B", bf))
        self._fh.write(eb); self._fh.write(sb); self._fh.write(tb)
        self._fh.write(data)

    def start_entry(self, name, type_str):
        if not name.startswith("/"): name = "/" + name
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

    def close(self): self._fh.close()


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
# Color classification
# ─────────────────────────────────────────────────────────────────────────────
COLOR_WHITE, COLOR_BLACK, COLOR_COLORED, COLOR_SILVER, COLOR_UNKNOWN = 0, 1, 2, 3, 4
_COLOR_NAMES = {0: "WHITE", 1: "BLACK", 2: "COLORED", 3: "SILVER", 4: "UNKNOWN"}


def classify_color(med_h, med_s, med_v):
    if med_v >= 180 and med_s < 60:  return COLOR_WHITE
    if med_v < 70:                   return COLOR_BLACK
    if med_s >= 80:                  return COLOR_COLORED
    return COLOR_SILVER


# ─────────────────────────────────────────────────────────────────────────────
# Kalman filter  (constant-velocity, 2-D field inches)
# ─────────────────────────────────────────────────────────────────────────────
class KalmanTrack:
    """
    State: [x, y, vx, vy]   Observation: [x, y]
    Constant-velocity model.  Pure numpy, no external dependency.
    """
    def __init__(self, np, x_in, y_in):
        self.np = np
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
        q = 4.0
        self.Q = np.diag([q, q, q*4, q*4])
        r = 3.0
        self.R = np.diag([r, r])
        self.x = np.array([[x_in],[y_in],[0.0],[0.0]], dtype=np.float64)
        self.P = np.diag([10., 10., 25., 25.])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0,0]), float(self.x[1,0])

    def update(self, x_meas, y_meas):
        np = self.np
        z = np.array([[x_meas],[y_meas]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def mahal_dist(self, x_obs, y_obs):
        """Mahalanobis distance from current state to an observation."""
        np = self.np
        z = np.array([[x_obs],[y_obs]])
        innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            d = float(innov.T @ np.linalg.inv(S) @ innov)
        except Exception:
            d = 1e9
        return math.sqrt(max(d, 0.0))

    @property
    def pos(self):
        return float(self.x[0,0]), float(self.x[1,0])

    @property
    def vel(self):
        return float(self.x[2,0]), float(self.x[3,0])


# ─────────────────────────────────────────────────────────────────────────────
# Appearance model — deep embeddings (MobileNetV3) or HSV histogram fallback
# ─────────────────────────────────────────────────────────────────────────────
class AppearanceModel:
    """
    Extracts a fixed-length feature vector from a robot blob crop.

    Primary: MobileNetV3-Small pretrained on ImageNet (torchvision).
             576-D features from the penultimate layer, L2-normalised.
             No fine-tuning required.  A white robot ends up in a completely
             different region of this embedding space from black/silver robots.

    Fallback: Full H×S×V histogram (12×4×4 = 192-D per region, two regions
              concatenated → 384-D), L2-normalised.  Captures brightness (V)
              unlike the old H×S-only approach.
    """

    def __init__(self, cv2, np, device="cpu"):
        self.cv2 = cv2
        self.np  = np
        self._model    = None
        self._preproc  = None
        self._device   = device
        self._use_deep = False
        self._try_init_deep()

    def _try_init_deep(self):
        torch = _try_import("torch")
        tv    = _try_import("torchvision")
        if torch is None or tv is None:
            print("[INFO] PyTorch not found — using HSV histogram ReID "
                  "(install torch+torchvision for significantly better tracking)")
            return
        try:
            import torchvision.models as models
            import torchvision.transforms as T
            net = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            net.classifier = torch.nn.Identity()
            net.eval()
            net = net.to(self._device)
            self._model = net
            self._preproc = T.Compose([
                T.ToPILImage(),
                T.Resize((64, 64)),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
            self._use_deep = True
            print("[INFO] MobileNetV3-Small loaded for deep ReID embeddings.")
        except Exception as e:
            print("[WARN] Could not load MobileNetV3: {} — using HSV fallback".format(e))

    def extract(self, frame, contour, cx_px, cy_px):
        """
        Returns (feat_vec, color_class, (med_h, med_s, med_v)) or None.
        feat_vec is L2-normalised.
        """
        cv2, np = self.cv2, self.np
        h_fr, w_fr = frame.shape[:2]
        x, y, bw, bh = cv2.boundingRect(contour)
        pad = 6
        x1 = max(0, x-pad); y1 = max(0, y-pad)
        x2 = min(w_fr, x+bw+pad); y2 = min(h_fr, y+bh+pad)
        if (x2-x1) < 12 or (y2-y1) < 12:
            return None

        crop = frame[y1:y2, x1:x2].copy()
        mask_full = np.zeros((h_fr, w_fr), dtype=np.uint8)
        cv2.drawContours(mask_full, [contour], -1, 255, -1)
        mask_crop = mask_full[y1:y2, x1:x2]

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        inner_r = max(2, int(min(bw, bh) * 0.08))
        inner_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (inner_r*2+1, inner_r*2+1))
        inner_mask = cv2.erode(mask_crop, inner_k)
        hsv_inner = hsv[inner_mask > 0]
        if len(hsv_inner) < 5:
            hsv_inner = hsv[mask_crop > 0]
        if len(hsv_inner) < 5:
            return None

        med_h = float(np.median(hsv_inner[:,0]))
        med_s = float(np.median(hsv_inner[:,1]))
        med_v = float(np.median(hsv_inner[:,2]))
        color_class = classify_color(med_h, med_s, med_v)

        if self._use_deep:
            try:
                import torch
                crop_masked = crop.copy()
                crop_masked[mask_crop == 0] = 0
                rgb = cv2.cvtColor(crop_masked, cv2.COLOR_BGR2RGB)
                inp = self._preproc(rgb).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    feat = self._model(inp).squeeze().cpu().numpy()
                nrm = float(np.linalg.norm(feat))
                if nrm > 1e-6: feat = feat / nrm
                return feat.astype(np.float32), color_class, (med_h, med_s, med_v)
            except Exception:
                pass

        # HSV histogram fallback (H×S×V, inner + ring regions)
        ring_mask = cv2.subtract(mask_crop, inner_mask)

        def _hist3(mask):
            hst = cv2.calcHist([hsv],[0,1,2], mask,
                               [12,4,4],[0,180,0,256,0,256]).flatten().astype(np.float32)
            s = float(hst.sum())
            return (hst/s) if s > 1e-6 else None

        ih = _hist3(inner_mask)
        rh = _hist3(ring_mask)
        if ih is None: return None
        if rh is None: rh = ih.copy()
        feat = np.concatenate([ih, rh])
        nrm = float(np.linalg.norm(feat))
        if nrm > 1e-6: feat = feat / nrm
        return feat.astype(np.float32), color_class, (med_h, med_s, med_v)

    def distance(self, feat_a, feat_b):
        """
        Returns distance in [0, 1].
        Deep: cosine distance (both are L2-normalised).
        Hist: Bhattacharyya-based distance.
        """
        if feat_a is None or feat_b is None: return 0.5
        np = self.np
        if self._use_deep:
            return float(max(0.0, 1.0 - np.dot(feat_a, feat_b)))
        else:
            bc = float(np.sum(np.sqrt(np.clip(feat_a * feat_b, 0, None))))
            bc = max(min(bc, 1.0), 1e-9)
            return min(1.0, -math.log(bc) / 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# Per-track identity state
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrackState:
    kalman      : object            = None
    feature     : Optional[object] = None
    color_class : int               = COLOR_UNKNOWN
    color_votes : object            = None
    n_updates   : int               = 0
    coast       : int               = 999
    pos_px      : Optional[Tuple]  = None
    area_history: List[float]       = dc_field(default_factory=list)

    COLOR_LOCK_N    = 15
    COLOR_LOCK_FRAC = 0.70

    def is_color_locked(self, np):
        if self.color_class == COLOR_UNKNOWN or self.color_votes is None:
            return False
        total = float(self.color_votes.sum())
        if total < self.COLOR_LOCK_N: return False
        return (float(self.color_votes[self.color_class]) / total
                >= self.COLOR_LOCK_FRAC)

    def update_color(self, cc, np):
        if self.color_votes is None:
            self.color_votes = np.zeros(5, dtype=np.float32)
        self.color_votes[cc] += 1.0
        self.color_class = int(np.argmax(self.color_votes))


# ─────────────────────────────────────────────────────────────────────────────
# RobotTracker  (v4.0)
# ─────────────────────────────────────────────────────────────────────────────
class RobotTracker:

    N_BG_SAMPLES  = 80
    FG_THRESH     = 25
    BLOB_MIN      = 300
    MIN_RADIUS_PX = 10
    KERNEL_PX     = 9
    MAX_COAST     = 60
    ROBOT_SIZE_IN = 18.0

    # Cost matrix weights
    W_MOTION = 0.40
    W_APPEAR = 0.45
    W_SIZE   = 0.15

    # Hard color-class mismatch penalty (added when locked classes differ)
    COLOR_MISMATCH_COST = 0.60

    # Mahalanobis gate: blobs beyond this are never assigned to a track
    MAHAL_GATE = 8.0

    # Appearance update: only blend when assignment cost is below this
    CONF_MAX_COST  = 0.55
    FEAT_EMA_ALPHA = 0.15

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
        self._tracks       = [TrackState() for _ in range(4)]
        self._initialized  = False
        self._appear       = None
        self._last_frame   = None

    # ── setup ────────────────────────────────────────────────────────────

    def setup(self, video_path, ordered_corners, frame_shape):
        cv2, np = self.cv2, self.np
        h, w = frame_shape[:2]

        tl, tr, br, bl = ordered_corners
        cx_poly = sum(p[0] for p in [tl,tr,br,bl])/4
        cy_poly = sum(p[1] for p in [tl,tr,br,bl])/4
        SIDE_PAD, BOTTOM_PAD, TOP_PAD = 25, 25, 5

        def _pad(x, y):
            dx=x-cx_poly; dy=y-cy_poly
            return (int(x+(SIDE_PAD if dx>0 else -SIDE_PAD)),
                    int(y+(BOTTOM_PAD if dy>0 else -TOP_PAD)))

        poly = np.array([_pad(*tl),_pad(*tr),_pad(*br),_pad(*bl)], dtype=np.int32)
        self._field_mask = np.zeros((h,w), dtype=np.uint8)
        cv2.fillPoly(self._field_mask, [poly], 255)

        dst2d = np.array([[0,144],[144,144],[144,0],[0,0]], dtype=np.float32)
        self._H_2d, _ = cv2.findHomography(ordered_corners, dst2d)
        self._H_inv   = np.linalg.inv(self._H_2d)

        c_f = np.array([[[72.,72.]]], dtype=np.float32)
        r_f = np.array([[[72.+self.ROBOT_SIZE_IN,72.]]], dtype=np.float32)
        u_f = np.array([[[72.,72.+self.ROBOT_SIZE_IN]]], dtype=np.float32)
        c_px = cv2.perspectiveTransform(c_f, self._H_inv)[0][0]
        r_px = cv2.perspectiveTransform(r_f, self._H_inv)[0][0]
        u_px = cv2.perspectiveTransform(u_f, self._H_inv)[0][0]
        self._robot_max_px = max(
            math.hypot(r_px[0]-c_px[0],r_px[1]-c_px[1]),
            math.hypot(u_px[0]-c_px[0],u_px[1]-c_px[1]))
        print("[INFO] Robot pixel footprint: {:.1f} px / 18 in".format(self._robot_max_px))

        neck_r = max(3, int(self._robot_max_px * 0.15))
        self._neck_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (neck_r*2+1, neck_r*2+1))

        # Median background
        cap2 = cv2.VideoCapture(video_path)
        n_total = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, n_total-1, self.N_BG_SAMPLES, dtype=int)
        bar = _make_bar("Building background", len(indices))
        frames = []
        for idx in indices:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, f = cap2.read()
            if ret: frames.append(f.astype(np.float32))
            bar.next()
        bar.finish(); cap2.release()
        self._bg = (np.median(frames,axis=0).astype(np.uint8)
                    if frames else np.zeros((h,w,3),dtype=np.uint8))
        print("[INFO] Background ready.")

        self._appear = AppearanceModel(cv2, np)

    # ── main update ───────────────────────────────────────────────────────

    def update(self, frame):
        cv2, np = self.cv2, self.np
        self._last_frame = frame

        if self._bg is None or self._H_2d is None:
            for p in self.tracked_poses: p.visible = False
            return self.tracked_poses

        blobs = self._get_blobs(frame)

        # ── init ──────────────────────────────────────────────────────────
        if not self._initialized:
            if len(blobs) < 4:
                for p in self.tracked_poses: p.visible = False
                return self.tracked_poses
            for i, b in enumerate(sorted(blobs[:4], key=lambda b: b[0])):
                fx, fy, _, _pid, cx, cy, cnt = b
                t = self._tracks[i]
                t.kalman = KalmanTrack(np, fx, fy)
                t.coast  = 0
                t.pos_px = (cx, cy)
                t.area_history = [float(cv2.contourArea(cnt))] * 5
                obs = self._appear.extract(frame, cnt, cx, cy)
                if obs is not None:
                    feat, cc, _ = obs
                    t.feature = feat.copy()
                    t.color_class = cc
                    t.color_votes = np.zeros(5, dtype=np.float32)
                    # Seed with strong confidence so lock triggers quickly
                    t.color_votes[cc] = float(TrackState.COLOR_LOCK_N)
                    t.n_updates = 5
                self.tracked_poses[i].x_in    = fx
                self.tracked_poses[i].y_in    = fy
                self.tracked_poses[i].visible = True
            self._initialized = True
            print("[INFO] Initialised. Colors: {}".format(
                [_COLOR_NAMES[self._tracks[k].color_class] for k in range(4)]))
            return self.tracked_poses

        # ── Kalman predict ────────────────────────────────────────────────
        for t in self._tracks:
            if t.kalman is not None:
                t.kalman.predict()

        # ── Pre-extract appearance for all blobs ──────────────────────────
        blob_obs = [self._appear.extract(frame, b[6], b[4], b[5]) for b in blobs]

        # ── Cost matrix ───────────────────────────────────────────────────
        n = len(blobs)
        INF  = 1e9
        SKIP = 2.0
        real_cost = np.full((4, max(n,1)), INF, dtype=np.float64)

        for i, t in enumerate(self._tracks):
            if t.kalman is None:
                real_cost[i,:] = SKIP * 0.9
                continue
            locked = t.is_color_locked(np)

            for j, b in enumerate(blobs):
                # Motion cost
                mahal = t.kalman.mahal_dist(b[0], b[1])
                if mahal > self.MAHAL_GATE:
                    real_cost[i,j] = INF
                    continue
                motion_cost = min(mahal / self.MAHAL_GATE, 1.0)

                # Appearance cost
                appear_cost = 0.5
                if t.feature is not None and blob_obs[j] is not None:
                    appear_cost = min(self._appear.distance(t.feature, blob_obs[j][0]), 1.0)

                # Size prior (soft tiebreaker)
                size_cost = 0.0
                if len(t.area_history) >= 3:
                    exp = float(np.median(t.area_history))
                    act = float(cv2.contourArea(b[6]))
                    if exp > 0:
                        size_cost = min(0.2, abs(math.log(max(act/exp, 1e-3))) * 0.15)

                # Hard color mismatch penalty
                color_cost = 0.0
                if locked and blob_obs[j] is not None:
                    bcc = blob_obs[j][1]
                    if bcc != COLOR_UNKNOWN and bcc != t.color_class:
                        color_cost = self.COLOR_MISMATCH_COST

                real_cost[i,j] = (self.W_MOTION * motion_cost +
                                  self.W_APPEAR * appear_cost +
                                  self.W_SIZE   * size_cost +
                                  color_cost)

        # Skip columns
        skip_cols = np.full((4,4), INF, dtype=np.float64)
        for i, t in enumerate(self._tracks):
            cf = min(t.coast / max(self.MAX_COAST,1), 1.0)
            skip_cols[i,i] = SKIP*(1.0 - 0.75*cf) if t.kalman else SKIP*0.9

        cost = np.hstack([real_cost, skip_cols])
        assignment = self._solve(cost, n)

        # ── Apply updates ─────────────────────────────────────────────────
        for i, t in enumerate(self._tracks):
            if i in assignment:
                j = assignment[i]
                fx, fy, hdg, _pid, cx, cy, cnt = blobs[j]
                blob_area = float(cv2.contourArea(cnt))

                if t.kalman is not None:
                    t.kalman.update(fx, fy)
                else:
                    t.kalman = KalmanTrack(np, fx, fy)

                t.coast  = 0
                t.pos_px = (cx, cy)
                t.area_history.append(blob_area)
                if len(t.area_history) > 20: t.area_history.pop(0)

                self.tracked_poses[i].x_in    = fx
                self.tracked_poses[i].y_in    = fy
                self.tracked_poses[i].heading = hdg
                self.tracked_poses[i].visible = True

                # Confident appearance / color update
                if float(real_cost[i,j]) < self.CONF_MAX_COST and blob_obs[j] is not None:
                    feat, cc, _ = blob_obs[j]
                    a = self.FEAT_EMA_ALPHA
                    if t.feature is None:
                        t.feature = feat.copy()
                    else:
                        t.feature = (1-a)*t.feature + a*feat
                        nrm = float(np.linalg.norm(t.feature))
                        if nrm > 1e-6: t.feature /= nrm

                    prev_class = t.color_class
                    t.update_color(cc, np)
                    t.n_updates = min(t.n_updates + 1, 500)

                    if (t.n_updates == TrackState.COLOR_LOCK_N
                            or (prev_class != t.color_class
                                and t.is_color_locked(np))):
                        print("[INFO] Track {} color locked: {}".format(
                            i, _COLOR_NAMES[t.color_class]))

            else:
                if not self._fallback_recover(i, frame):
                    t.coast = min(t.coast + 1, self.MAX_COAST + 1)
                    self.tracked_poses[i].visible = (
                        t.coast <= self.MAX_COAST and t.kalman is not None)
                    if t.kalman is not None:
                        px, py = t.kalman.pos
                        self.tracked_poses[i].x_in = px
                        self.tracked_poses[i].y_in = py

        return self.tracked_poses

    # ── assignment ────────────────────────────────────────────────────────

    def _solve(self, cost, n_blobs):
        np = self.np
        INF = 1e9
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
            return {r: c for r, c in zip(row_ind, col_ind) if c < n_blobs}
        except ImportError:
            pass
        assignment = {}
        work = cost.copy()
        for _ in range(4):
            idx = int(np.argmin(work))
            ri = idx // (n_blobs+4); ci = idx % (n_blobs+4)
            if work[ri,ci] >= INF: break
            if ci < n_blobs: assignment[ri] = ci
            work[ri,:] = INF; work[:,ci] = INF
        return assignment

    # ── fallback recovery ─────────────────────────────────────────────────

    def _fallback_recover(self, i, frame):
        cv2, np = self.cv2, self.np
        t = self._tracks[i]
        if t.kalman is None or self._bg is None: return False

        px_r, py_r = t.kalman.pos
        diff2 = cv2.absdiff(frame, self._bg)
        gray2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2GRAY)
        _, fg2 = cv2.threshold(gray2, max(8, self.FG_THRESH//3), 255, cv2.THRESH_BINARY)
        fg2 = cv2.bitwise_and(fg2, self._field_mask)
        fg2 = cv2.morphologyEx(fg2, cv2.MORPH_CLOSE, self._kern)
        cnts2, _ = cv2.findContours(fg2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_d, best_blob = self.MAHAL_GATE * 1.5, None
        for c2 in cnts2:
            if cv2.contourArea(c2) < self.BLOB_MIN // 2: continue
            M2 = cv2.moments(c2)
            if M2["m00"] == 0: continue
            bx2 = M2["m10"]/M2["m00"]; by2 = M2["m01"]/M2["m00"]
            pt2 = np.array([[[bx2,by2]]], dtype=np.float32)
            fp2 = cv2.perspectiveTransform(pt2, self._H_2d)[0][0]
            fx2, fy2 = float(fp2[0]), float(fp2[1])
            if not (0 <= fx2 <= 144 and 0 <= fy2 <= 144): continue
            d2 = math.hypot(fx2-px_r, fy2-py_r)
            if d2 < best_d:
                best_d = d2
                best_blob = (fx2, fy2, int(bx2), int(by2))

        if best_blob is None: return False
        fx2, fy2, cx2, cy2 = best_blob
        t.kalman.update(fx2, fy2)
        t.pos_px = (cx2, cy2)
        self.tracked_poses[i].x_in = fx2
        self.tracked_poses[i].y_in = fy2
        self.tracked_poses[i].visible = True
        return True

    # ── foreground ────────────────────────────────────────────────────────

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
        return sorted([c for c in cnts if cv2.contourArea(c) >= self.BLOB_MIN],
                      key=lambda c: -cv2.contourArea(c))

    def _get_blobs(self, frame):
        """
        Returns (fx_in, fy_in, heading, parent_id, cx_px, cy_px, contour) list.
        Uses Kalman-predicted positions as watershed seeds for merged blobs.
        """
        cv2, np = self.cv2, self.np
        fg = self._foreground_mask(frame)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for pid, c in enumerate(cnts):
            area = float(cv2.contourArea(c))
            if area < self.BLOB_MIN: continue
            bm = np.zeros(self._field_mask.shape, dtype=np.uint8)
            cv2.drawContours(bm, [c], -1, 255, -1)
            dr = cv2.distanceTransform(bm, cv2.DIST_L2, 5)
            if dr.max() < self.MIN_RADIUS_PX: continue

            x, y, bw, bh = cv2.boundingRect(c)
            seeds = None
            if max(bw,bh) >= self._robot_max_px * 1.6 and self._initialized:
                seeds = []
                for t in self._tracks:
                    if t.kalman is None: continue
                    pxf, pyf = t.kalman.pos
                    pt = np.array([[[pxf,pyf]]], dtype=np.float32)
                    ip = cv2.perspectiveTransform(pt, self._H_inv)[0][0]
                    if cv2.pointPolygonTest(c,(float(ip[0]),float(ip[1])),False) >= 0:
                        seeds.append((float(ip[0]),float(ip[1])))
                if len(seeds) < 2: seeds = None

            subs = self._split_contour(c, seeds)
            if not subs: continue
            per_area = area / len(subs)
            for cx, cy in subs:
                pt = np.array([[[cx,cy]]], dtype=np.float32)
                fp = cv2.perspectiveTransform(pt, self._H_2d)[0][0]
                fx, fy = float(fp[0]), float(fp[1])
                if not (0 <= fx <= 144 and 0 <= fy <= 144): continue
                candidates.append((fx, fy, 0.0, per_area, pid, int(cx), int(cy), c))

        candidates.sort(key=lambda b: -b[3])
        return [(fx,fy,hdg,pid,cx,cy,cnt)
                for fx,fy,hdg,_a,pid,cx,cy,cnt in candidates]

    def _split_contour(self, contour, seeds=None):
        cv2, np = self.cv2, self.np
        h, w = self._field_mask.shape[:2]
        mask = np.zeros((h,w), dtype=np.uint8)
        cv2.drawContours(mask,[contour],-1,255,-1)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist.max() == 0: return []

        _, pm = cv2.threshold(dist, dist.max()*0.40, 255, cv2.THRESH_BINARY)
        pm = cv2.erode(pm.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        n_labels, labels = cv2.connectedComponents(pm)
        centers = []
        for lbl in range(1, n_labels):
            M = cv2.moments((labels==lbl).astype(np.uint8))
            if M["m00"] > 0:
                centers.append((M["m10"]/M["m00"], M["m01"]/M["m00"]))

        if (len(centers) < 2 and seeds is not None and len(seeds) == 2
                and self._robot_max_px > 0):
            x,y,bw,bh = cv2.boundingRect(contour)
            if max(bw,bh) >= self._robot_max_px * 1.4:
                ws = self._watershed_split(mask, seeds)
                if len(ws) == 2: return ws
        return centers

    def _watershed_split(self, binary_mask, seeds):
        cv2, np = self.cv2, self.np
        h, w = binary_mask.shape[:2]
        img3 = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        markers = np.zeros((h,w), dtype=np.int32)
        seed_r = max(4, int(self._robot_max_px * 0.1))
        for lbl,(sx,sy) in enumerate(seeds, start=1):
            sx,sy = int(sx),int(sy)
            if 0<=sx<w and 0<=sy<h:
                cv2.circle(markers,(sx,sy),seed_r,lbl,-1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        outer = cv2.bitwise_and(cv2.dilate(binary_mask,k,iterations=2),
                                cv2.bitwise_not(binary_mask))
        markers[outer>0] = 3
        if not (np.any(markers==1) and np.any(markers==2)): return []
        cv2.watershed(img3, markers)
        centers = []
        for lbl in [1,2]:
            M = cv2.moments((markers==lbl).astype(np.uint8))
            if M["m00"]>0:
                centers.append((M["m10"]/M["m00"],M["m01"]/M["m00"]))
        return centers if len(centers)==2 else []


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
            cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(5,5),0),30,100)
        cnts, _ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
            if cv2.contourArea(cnt) < h*w*0.05: continue
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt,True), True)
            if len(approx)==4:
                pts = approx.reshape(4,2).astype(np.float32)
                s = pts.sum(axis=1); diff = np.diff(pts,axis=1).flatten()
                ordered = np.array([pts[np.argmin(s)],pts[np.argmin(diff)],
                                    pts[np.argmax(s)],pts[np.argmax(diff)]],dtype=np.float32)
                H, _ = cv2.findHomography(ordered,
                    np.array([[0,144],[144,144],[144,0],[0,0]],np.float32))
                return ordered, H
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ─────────────────────────────────────────────────────────────────────────────
def process_match(video_path, output_dir, start_offset_sec=0.0,
                  sample_rate_fps=10.0, debug=False, debug_every_n=10,
                  manual_corners_px=None):
    cv2 = _require("cv2", "opencv-python")
    np  = _require("numpy")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print("[ERROR] Cannot open:", video_path); sys.exit(1)

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("[INFO] Video: {:.1f} fps, {} frames, {:.1f}s".format(
        video_fps, total_frames, total_frames/video_fps))

    start_frame = int(start_offset_sec * video_fps)
    frame_step  = max(1, int(round(video_fps/sample_rate_fps)))
    print("[INFO] Processing every {} frames (~{:.1f} fps output)".format(
        frame_step, video_fps/frame_step))

    csv_path    = os.path.join(output_dir, "robot_positions.csv")
    wpilog_path = os.path.join(output_dir, "match_log.wpilog")
    debug_dir   = os.path.join(output_dir, "tracker_debug") if debug else None
    if debug_dir: os.makedirs(debug_dir, exist_ok=True)

    log       = WPILogWriter(wpilog_path)
    pose_eids = [log.start_entry("Robot{}/Pose".format(i),"double[]") for i in range(4)]
    vis_eids  = [log.start_entry("Robot{}/Visible".format(i),"boolean") for i in range(4)]

    csv_file = open(csv_path,"w",newline="")
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
    ordered = None; H_inv = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, sample_frame = cap.read()
    if not ret: print("[ERROR] Could not read first frame."); sys.exit(1)

    if manual_corners_px is not None:
        bl,br,tr,tl = [np.array(c,np.float32) for c in manual_corners_px]
        ordered = np.array([tl,tr,br,bl],np.float32)
        tracker.setup(video_path, ordered, sample_frame.shape)
        H_inv = tracker._H_inv
        print("[INFO] Manual corners loaded.")
        if tracker._bg is not None:
            cv2.imwrite(os.path.join(output_dir,"median_background.jpg"),tracker._bg)
    else:
        print("[WARN] No corners — will attempt auto-detection.")

    frame_num = start_frame; processed = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame if start_frame > 0 else 0)
    debug_colors = [(220,160,0),(0,220,255),(0,0,220),(0,120,255)]
    n_to_process = max(1,(total_frames-start_frame+frame_step-1)//frame_step)
    bar = _make_bar("Processing frames", n_to_process)

    while True:
        ret, frame = cap.read()
        if not ret: break

        match_time_s = frame_num/video_fps - start_offset_sec
        timestamp_us = max(0, int(match_time_s*1_000_000))
        frame_num += 1

        if (frame_num-start_frame) % frame_step != 0: continue
        bar.next()

        if ordered is None:
            result = field_detector.detect_field(frame)
            if result is not None:
                ordered, _ = result
                tracker.setup(video_path, ordered, frame.shape)
                H_inv = tracker._H_inv
                print("\n  [t={:.1f}s] Field auto-detected.".format(match_time_s))
                if tracker._bg is not None:
                    cv2.imwrite(os.path.join(output_dir,"median_background.jpg"),tracker._bg)

        poses = tracker.update(frame)

        for i, p in enumerate(poses):
            log.write_pose2d(pose_eids[i], timestamp_us, p.x_m, p.y_m, p.heading)
            log.write_boolean(vis_eids[i], timestamp_us, p.visible)

        row = ["{:.4f}".format(match_time_s)]
        for p in poses:
            row += ["{:.2f}".format(p.x_in),"{:.2f}".format(p.y_in),
                    "{:.4f}".format(p.heading),"1" if p.visible else "0"]
        csv_writer.writerow(row)

        if debug and debug_dir and H_inv is not None:
            dbg = frame.copy()
            if ordered is not None:
                cv2.polylines(dbg,[ordered.astype(np.int32).reshape(-1,1,2)],True,(0,200,0),2)
            for c in tracker._foreground_contours(frame):
                bm = np.zeros(tracker._field_mask.shape,np.uint8)
                cv2.drawContours(bm,[c],-1,255,-1)
                dr = cv2.distanceTransform(bm,cv2.DIST_L2,5)
                col = (0,255,255) if dr.max()>=tracker.MIN_RADIUS_PX else (160,160,160)
                cv2.drawContours(dbg,[c],-1,col,2)

            fh,fw = frame.shape[:2]; n_drawn = 0
            for i,p in enumerate(poses):
                if not p.visible: continue
                pt  = np.array([[[p.x_in,p.y_in]]],dtype=np.float32)
                img = cv2.perspectiveTransform(pt, H_inv)[0][0]
                ix,iy = int(img[0]),int(img[1])
                if not (0<=ix<fw and 0<=iy<fh): continue
                col = debug_colors[i]
                cv2.circle(dbg,(ix,iy),16,col,-1)
                cv2.circle(dbg,(ix,iy),19,(255,255,255),2)
                t = tracker._tracks[i]
                cc = _COLOR_NAMES.get(t.color_class,"?")
                lk = "L" if t.is_color_locked(np) else ""
                cv2.putText(dbg,"R{} {}{}".format(i,cc[:3],lk),
                            (ix-14,iy-24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
                if t.kalman:
                    vx,vy = t.kalman.vel
                    if abs(vx)+abs(vy)>0.5:
                        pt2 = np.array([[[p.x_in+vx*3,p.y_in+vy*3]]],dtype=np.float32)
                        im2 = cv2.perspectiveTransform(pt2,H_inv)[0][0]
                        cv2.arrowedLine(dbg,(ix,iy),(int(im2[0]),int(im2[1])),(255,255,255),2)
                n_drawn += 1

            cv2.putText(dbg,"t={:.2f}s [{}/4]".format(match_time_s,n_drawn),
                        (12,32),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
            mode = "DEEP" if (tracker._appear and tracker._appear._use_deep) else "HIST"
            cv2.putText(dbg,"ReID:{}".format(mode),(12,60),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,0),1)
            if processed % debug_every_n == 0:
                cv2.imwrite(os.path.join(debug_dir,"frame_{:06d}.jpg".format(processed)),dbg)

        processed += 1

    bar.finish()
    log.close(); csv_file.close(); cap.release()
    print("\n[DONE] {} frames processed.".format(processed))
    print("  CSV:    {}".format(csv_path))
    print("  WPILOG: {}".format(wpilog_path))
    if debug: print("  Debug:  {}".format(debug_dir))
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
|  Debug: track label + color class (WHI/BLK/COL/SIL)             |
|  "L" = color class locked. Arrow = Kalman velocity vector.       |
|  "ReID:DEEP" = MobileNetV3 active. "ReID:HIST" = HSV fallback.  |
+------------------------------------------------------------------+
""")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def download_video(url, output_dir):
    import subprocess
    out = os.path.join(output_dir,"match_video.mp4")
    print("[INFO] Downloading:", url)
    bar = _make_bar("Downloading",100); last_pct = 0
    proc = subprocess.Popen(
        ["yt-dlp","-f","best[ext=mp4]","-o",out,url,
         "--progress-template","%(progress._percent_str)s"],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if proc.stdout:
        for line in proc.stdout:
            try:
                pct = int(float(line.strip().rstrip("%")))
                while last_pct<pct: bar.next(); last_pct+=1
            except ValueError: pass
    proc.wait()
    while last_pct<100: bar.next(); last_pct+=1
    bar.finish()
    if proc.returncode!=0: print("[ERROR] yt-dlp failed"); sys.exit(1)
    return out


def main():
    p = argparse.ArgumentParser(description="Track FTC DECODE robots (v4.0)")
    p.add_argument("url",            nargs="?")
    p.add_argument("--output-dir",   default="./output")
    p.add_argument("--start-offset", type=float, default=0.0)
    p.add_argument("--sample-rate",  type=float, default=10.0)
    p.add_argument("--debug",        action="store_true")
    p.add_argument("--debug-every",  type=int,   default=5)
    p.add_argument("--no-download",  action="store_true")
    p.add_argument("--video-path",   default=None)
    p.add_argument("--corners",      default=None)
    args = p.parse_args()

    if args.no_download:
        if not args.video_path: p.error("--no-download requires --video-path")
        video_path = args.video_path
    else:
        if not args.url: p.error("YouTube URL required (or --no-download --video-path)")
        os.makedirs(args.output_dir, exist_ok=True)
        video_path = download_video(args.url, args.output_dir)

    corners = None
    if args.corners:
        with open(args.corners) as f: corners = json.load(f)["corners_px"]

    process_match(video_path=video_path, output_dir=args.output_dir,
                  start_offset_sec=args.start_offset, sample_rate_fps=args.sample_rate,
                  debug=args.debug, debug_every_n=args.debug_every,
                  manual_corners_px=corners)


if __name__ == "__main__":
    main()