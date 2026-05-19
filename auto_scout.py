import argparse
import builtins
import csv
import itertools
import json
import math
import os
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from typing import Dict, FrozenSet, List, Optional, Tuple

from util.juice_log import CSV_COLUMNS, ROBOT_POSE_SCHEMA, JuiceLogWriter, csv_row_to_list, read_rows as read_jlog_rows, sniff_jlog

SAVE_ALL_DEBUG_AROUND_MERGES = False
PROCESS_EVERY_SOURCE_FRAME = True
FIELD_SIZE_IN = 144.0
FIELD_CENTER_OFFSET_IN = FIELD_SIZE_IN / 2.0
ANSI_RESET = "\033[0m"
ANSI_STYLES = {
    "info": "\033[1;36m",
    "warn": "\033[1;33m",
    "error": "\033[1;31m",
    "done": "\033[1;32m",
    "path": "\033[0;94m",
    "shot_made": "\033[1;32m",
    "shot_missed": "\033[1;31m",
}
COLOR_OUTPUT_ENABLED = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_ACTIVE_PROGRESS_BAR = None


def _ansi(style: str, text: str) -> str:
    if not COLOR_OUTPUT_ENABLED:
        return text
    code = ANSI_STYLES.get(style)
    if not code:
        return text
    return "{}{}{}".format(code, text, ANSI_RESET)


def _style_console_text(text: str) -> str:
    if not COLOR_OUTPUT_ENABLED or not isinstance(text, str):
        return text

    stripped = text.lstrip("\r\n")
    prefix = text[:len(text) - len(stripped)]

    tag_styles = {
        "[INFO]": "info",
        "[WARN]": "warn",
        "[ERROR]": "error",
        "[DONE]": "done",
    }
    for tag, style in tag_styles.items():
        if stripped.startswith(tag):
            stripped = _ansi(style, tag) + stripped[len(tag):]
            break

    if "Shot made by" in stripped:
        stripped = stripped.replace("Shot made by", "{} made by".format(
            _ansi("shot_made", "Shot")), 1)
    elif "Shot missed by" in stripped:
        stripped = stripped.replace("Shot missed by", "{} missed by".format(
            _ansi("shot_missed", "Shot")), 1)

    if stripped.startswith("  CSV:"):
        stripped = "  CSV:    {}".format(_ansi("path", stripped.split("CSV:", 1)[1].strip()))
    elif stripped.startswith("  JLOG:"):
        stripped = "  JLOG:   {}".format(_ansi("path", stripped.split("JLOG:", 1)[1].strip()))
    elif stripped.startswith("  WPILOG:"):
        stripped = "  WPILOG: {}".format(_ansi("path", stripped.split("WPILOG:", 1)[1].strip()))
    elif stripped.startswith("  Debug:"):
        stripped = "  Debug:  {}".format(_ansi("path", stripped.split("Debug:", 1)[1].strip()))

    return prefix + stripped


def _console_print(*args, **kwargs):
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    file = kwargs.pop("file", None)
    flush = kwargs.pop("flush", False)
    if kwargs:
        raise TypeError("Unsupported print kwargs: {}".format(", ".join(sorted(kwargs.keys()))))
    if file is None:
        file = sys.stdout
    text = sep.join(str(arg) for arg in args)
    if file in (sys.stdout, sys.stderr):
        text = _style_console_text(text)
    if _ACTIVE_PROGRESS_BAR is not None and file in (sys.stdout, sys.stderr):
        _clear_active_progress_bar()
    builtins.print(text, end=end, file=file, flush=flush)
    if _ACTIVE_PROGRESS_BAR is not None and file in (sys.stdout, sys.stderr):
        _redraw_active_progress_bar()


print = _console_print


def _clear_active_progress_bar():
    bar = _ACTIVE_PROGRESS_BAR
    if bar is None:
        return
    file = getattr(bar, "file", None)
    is_tty = getattr(bar, "is_tty", None)
    if file is None or not callable(is_tty) or not is_tty():
        return
    width = max(0, int(getattr(bar, "_max_width", 0)))
    if width > 0:
        builtins.print("\r{}\r".format(" " * width), end="", file=file, flush=True)
    else:
        builtins.print("\r", end="", file=file, flush=True)


def _redraw_active_progress_bar():
    bar = _ACTIVE_PROGRESS_BAR
    if bar is None:
        return
    update = getattr(bar, "update", None)
    if callable(update):
        update()


class _ManagedBar:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def next(self, n=1):
        return self._inner.next(n)

    def update(self):
        return self._inner.update()

    def finish(self):
        global _ACTIVE_PROGRESS_BAR
        try:
            return self._inner.finish()
        finally:
            if _ACTIVE_PROGRESS_BAR is self:
                _ACTIVE_PROGRESS_BAR = None


def _require(package, pip_name=None):
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        name = pip_name or package
        print("[ERROR] Missing: {}\n  Install: pip install {}".format(name, name))
        sys.exit(1)


def _make_bar(label, max_val):
    global _ACTIVE_PROGRESS_BAR
    try:
        from progress.bar import Bar
        bar = _ManagedBar(Bar(
            label,
            max=max_val,
            suffix="%(percent).0f%% %(elapsed_td)s ETA %(eta_td)s",
        ))
        _ACTIVE_PROGRESS_BAR = bar
        return bar
    except ImportError:
        class _FallbackBar:
            def __init__(self, lbl, total):
                self._lbl = lbl
                self._total = max(total, 1)
                self._n = 0
                self.file = sys.stdout
                self._max_width = 0
                self._render()

            def is_tty(self):
                return self.file.isatty()

            def _render(self):
                pct = int(self._n / self._total * 100)
                line = "[{}] {}%".format(self._lbl, pct)
                self._max_width = max(self._max_width, len(line))
                builtins.print("\r{}".format(line), end="", file=self.file, flush=True)

            def next(self):
                self._n += 1
                if self._n % max(1, self._total // 20) == 0 or self._n == self._total:
                    self._render()

            def update(self):
                self._render()

            def finish(self):
                builtins.print(file=self.file, flush=True)

        bar = _ManagedBar(_FallbackBar(label, max_val))
        _ACTIVE_PROGRESS_BAR = bar
        return bar


def _field_corner_to_center_xy(x_in: float, y_in: float) -> Tuple[float, float]:
    return float(x_in) - FIELD_CENTER_OFFSET_IN, float(y_in) - FIELD_CENTER_OFFSET_IN


def _field_center_to_corner_xy(x_in: float, y_in: float) -> Tuple[float, float]:
    return float(x_in) + FIELD_CENTER_OFFSET_IN, float(y_in) + FIELD_CENTER_OFFSET_IN


def _normalize_angle_rad(angle: float) -> float:
    while angle <= -math.pi:
        angle += math.tau
    while angle > math.pi:
        angle -= math.tau
    return angle


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
    x_in:    float = FIELD_CENTER_OFFSET_IN
    y_in:    float = FIELD_CENTER_OFFSET_IN
    heading: float = 0.0
    visible: bool  = False

    @property
    def x_center_in(self): return self.x_in - FIELD_CENTER_OFFSET_IN

    @property
    def y_center_in(self): return self.y_in - FIELD_CENTER_OFFSET_IN

    @property
    def x_m(self): return self.x_center_in * 0.0254

    @property
    def y_m(self): return self.y_center_in * 0.0254

    @property
    def wpilog_x_m(self): return self.y_center_in * 0.0254

    @property
    def wpilog_y_m(self): return self.x_center_in * 0.0254

    @property
    def wpilog_heading_rad(self): return _normalize_angle_rad((math.pi / 2.0) - self.heading)


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
    order_votes    : dict                  = dc_field(default_factory=dict)
    entry_features : dict                  = dc_field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Shot detection
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ShotEvent:
    shooter_id: int
    result: str
    shot_x_in: float
    shot_y_in: float
    frame_num: int
    timestamp_s: float
    goal_color: str = ""


@dataclass
class BallTrack:
    track_id: int
    samples: List[Tuple[int, float, float]] = dc_field(default_factory=list)
    missing_frames: int = 0
    launched: bool = False
    shooter_id: Optional[int] = None
    shooter_img: Optional[Tuple[float, float]] = None
    shooter_dist_px: Optional[float] = None
    shooter_pos_center_in: Optional[Tuple[float, float]] = None
    goal_color: str = ""
    entered_goal: bool = False
    entered_goal_frames: int = 0
    first_goal_entry_frame: Optional[int] = None
    last_goal_entry_point: Optional[Tuple[float, float]] = None
    approached_goal: bool = False
    resolved: bool = False

    @property
    def last_point(self) -> Optional[Tuple[int, float, float]]:
        return self.samples[-1] if self.samples else None

    @property
    def first_point(self) -> Optional[Tuple[int, float, float]]:
        return self.samples[0] if self.samples else None


class ShotDetector:
    BALL_MIN_CONTOUR_AREA_PX = 8.0
    BALL_MAX_CONTOUR_AREA_PX = 420.0
    BALL_ASSOC_DIST_PX = 34.0
    BALL_ASSOC_DIST_FAST_PX = 124.0
    BALL_MAX_MISSING = 12
    SHOT_MAX_START_DIST_PX = 84.0
    SHOT_MIN_DISP_PX = 8.0
    SHOT_MIN_SPEED_PX = 2.0
    SHOT_MIN_UPWARD_PX = 3.0
    SHOT_MIN_LIFE_FRAMES = 2
    SHOT_MIN_NEGATIVE_STEP_RATIO = 0.35
    SHOT_MIN_AWAY_FROM_SHOOTER_PX = 2.0
    SHOT_DEDUP_TIME_S = 0.35
    GOAL_OPENING_TOP_FRAC = 0.58
    GOAL_APPROACH_SIDE_PAD_FRAC = 0.30
    GOAL_APPROACH_DOWN_FRAC = 0.95
    GOAL_CONFIRM_MIN_FRAMES = 2
    GOAL_ENTRY_MAX_MISSING = 5
    GOAL_OPENING_SHRINK_X = 0.84
    GOAL_OPENING_SHRINK_Y = 0.80
    GOAL_BLUE_HSV_LO = (95, 80, 45)
    GOAL_BLUE_HSV_HI = (135, 255, 255)
    GOAL_RED1_HSV_LO = (0, 90, 45)
    GOAL_RED1_HSV_HI = (12, 255, 255)
    GOAL_RED2_HSV_LO = (170, 90, 45)
    GOAL_RED2_HSV_HI = (179, 255, 255)
    GOAL_MIN_AREA_PX = 900.0

    def __init__(self, cv2, np):
        self.cv2, self.np = cv2, np
        self._next_track_id = 1
        self._tracks: Dict[int, BallTrack] = {}
        self._goal_openings: Dict[str, object] = {}
        self._goal_approaches: Dict[str, object] = {}
        self._last_event_time_by_robot: Dict[int, float] = {}
        self._ready = False

    def setup(self, tracker) -> None:
        if tracker._bg is None or tracker._field_mask is None:
            return
        openings = self._detect_goal_openings(tracker._bg, tracker._field_mask)
        if not openings:
            openings = self._fallback_goal_openings(tracker)
        if not openings:
            print("[WARN] Shot detector could not determine goal openings.")
            return
        self._goal_openings = openings
        self._goal_approaches = {
            color: self._build_goal_approach(poly)
            for color, poly in openings.items()
        }
        self._ready = True
        print("[INFO] Shot detector goal openings ready: {}".format(
            ", ".join(sorted(openings.keys()))))

    def update(self, frame_num: int, match_time_s: float, poses, tracker) -> List[ShotEvent]:
        if not self._ready or tracker._last_ball_mask is None or tracker._H_inv is None:
            return []

        detections = self._extract_ball_detections(tracker._last_ball_mask)
        robot_refs = self._robot_refs(poses, tracker)
        self._associate_tracks(detections, robot_refs, frame_num)

        events = []
        for track in list(self._tracks.values()):
            if not track.samples or track.resolved:
                continue
            self._update_goal_state(track)
            self._maybe_mark_launched(track, robot_refs)
            if track.missing_frames > self.BALL_MAX_MISSING:
                event = self._resolve_track(track, frame_num, match_time_s)
                if event is not None and not self._is_duplicate_event(event):
                    events.append(event)
                del self._tracks[track.track_id]
        return events

    def draw_debug(self, dbg) -> None:
        if not self._ready:
            return
        cv2 = self.cv2
        for color, poly in self._goal_approaches.items():
            cv_color = (255, 180, 60) if color == "blue" else (80, 180, 255)
            cv2.polylines(dbg, [poly], True, cv_color, 1, cv2.LINE_AA)
        for color, poly in self._goal_openings.items():
            cv_color = (255, 80, 0) if color == "blue" else (0, 80, 255)
            cv2.polylines(dbg, [poly], True, cv_color, 2, cv2.LINE_AA)
        for track in self._tracks.values():
            pts = [(int(round(x)), int(round(y))) for _f, x, y in track.samples[-8:]]
            if len(pts) >= 2:
                cv2.polylines(
                    dbg,
                    [self.np.array(pts, dtype=self.np.int32).reshape(-1, 1, 2)],
                    False,
                    (0, 255, 255) if track.launched else (180, 180, 180),
                    2,
                    cv2.LINE_AA,
                )
            if pts:
                px, py = pts[-1]
                label = "S{}".format(track.shooter_id) if track.launched and track.shooter_id is not None else "b"
                if track.entered_goal:
                    label += ":goal"
                cv2.putText(dbg, label, (px + 4, py - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _extract_ball_detections(self, ball_mask):
        cv2 = self.cv2
        cnts, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < self.BALL_MIN_CONTOUR_AREA_PX or area > self.BALL_MAX_CONTOUR_AREA_PX:
                continue
            M = cv2.moments(c)
            if M["m00"] <= 0:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            detections.append((cx, cy, area))
        return detections

    def _robot_refs(self, poses, tracker):
        refs = []
        for rid, pose in enumerate(poses):
            if not pose.visible:
                continue
            pt = self.np.array([[[pose.x_in, pose.y_in]]], dtype=self.np.float32)
            ip = self.cv2.perspectiveTransform(pt, tracker._H_inv)[0][0]
            refs.append({
                "id": rid,
                "img": (float(ip[0]), float(ip[1])),
                "center": (float(pose.x_center_in), float(pose.y_center_in)),
            })
        return refs

    def _associate_tracks(self, detections, robot_refs, frame_num: int) -> None:
        unmatched_tracks = set(self._tracks.keys())
        unmatched_dets = set(range(len(detections)))
        pairs = []
        for tid, track in self._tracks.items():
            pred_x, pred_y = self._predict_track_point(track)
            max_dist = self.BALL_ASSOC_DIST_FAST_PX if track.launched else self.BALL_ASSOC_DIST_PX
            for di, (cx, cy, _area) in enumerate(detections):
                d = math.hypot(cx - pred_x, cy - pred_y)
                if d <= max_dist:
                    pairs.append((d, tid, di))
        pairs.sort()
        for _d, tid, di in pairs:
            if tid not in unmatched_tracks or di not in unmatched_dets:
                continue
            cx, cy, _area = detections[di]
            track = self._tracks[tid]
            track.samples.append((frame_num, cx, cy))
            track.missing_frames = 0
            unmatched_tracks.remove(tid)
            unmatched_dets.remove(di)

        for tid in unmatched_tracks:
            self._tracks[tid].missing_frames += 1

        for di in unmatched_dets:
            cx, cy, _area = detections[di]
            nearest_id = None
            nearest_dist = None
            nearest_img = None
            nearest_center = None
            for ref in robot_refs:
                dist = math.hypot(cx - ref["img"][0], cy - ref["img"][1])
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest_id = ref["id"]
                    nearest_img = ref["img"]
                    nearest_center = ref["center"]
            self._tracks[self._next_track_id] = BallTrack(
                track_id=self._next_track_id,
                samples=[(frame_num, cx, cy)],
                shooter_id=nearest_id,
                shooter_img=nearest_img,
                shooter_dist_px=nearest_dist,
                shooter_pos_center_in=nearest_center,
            )
            self._next_track_id += 1

    def _predict_track_point(self, track: BallTrack) -> Tuple[float, float]:
        if len(track.samples) < 2:
            _f, x, y = track.samples[-1]
            return x, y
        _f1, x1, y1 = track.samples[-1]
        _f0, x0, y0 = track.samples[-2]
        return x1 + (x1 - x0), y1 + (y1 - y0)

    def _update_goal_state(self, track: BallTrack) -> None:
        frame_idx, x, y = track.samples[-1]
        for color, poly in self._goal_approaches.items():
            if self.cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                track.approached_goal = True
                if not track.goal_color:
                    track.goal_color = color
        for color, poly in self._goal_openings.items():
            if self.cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                track.entered_goal = True
                track.entered_goal_frames += 1
                if track.first_goal_entry_frame is None:
                    track.first_goal_entry_frame = frame_idx
                track.last_goal_entry_point = (float(x), float(y))
                track.goal_color = color
                break

    def _maybe_mark_launched(self, track: BallTrack, robot_refs) -> None:
        if track.launched or len(track.samples) < self.SHOT_MIN_LIFE_FRAMES:
            return
        first = track.first_point
        last = track.last_point
        if first is None or last is None:
            return
        shooter_ref, shooter_score = self._select_shooter(track, robot_refs)
        if shooter_ref is None or shooter_score > self.SHOT_MAX_START_DIST_PX:
            return
        f0, x0, y0 = first
        f1, x1, y1 = last
        dt = max(f1 - f0, 1)
        disp = math.hypot(x1 - x0, y1 - y0)
        avg_speed = disp / dt
        if disp < self.SHOT_MIN_DISP_PX or avg_speed < self.SHOT_MIN_SPEED_PX:
            return
        if (y1 - y0) > -self.SHOT_MIN_UPWARD_PX:
            return
        if not self._has_consistent_upward_motion(track):
            return
        if not self._moving_away_from_shooter(track):
            return
        track.shooter_id = shooter_ref["id"]
        track.shooter_img = shooter_ref["img"]
        track.shooter_dist_px = shooter_score
        track.shooter_pos_center_in = shooter_ref["center"]
        track.launched = True

    def _resolve_track(self, track: BallTrack, frame_num: int, match_time_s: float) -> Optional[ShotEvent]:
        if not track.launched or track.shooter_id is None or track.shooter_pos_center_in is None:
            return None
        if self._confirmed_goal_make(track):
            result = "made"
        elif track.approached_goal or self._track_reached_top(track):
            result = "missed"
        else:
            return None
        sx, sy = track.shooter_pos_center_in
        track.resolved = True
        return ShotEvent(
            shooter_id=track.shooter_id,
            result=result,
            shot_x_in=float(sx),
            shot_y_in=float(sy),
            frame_num=frame_num,
            timestamp_s=match_time_s,
            goal_color=track.goal_color,
        )

    def _has_consistent_upward_motion(self, track: BallTrack) -> bool:
        if len(track.samples) < 3:
            return False
        negative_steps = 0
        total_steps = 0
        for (_f0, _x0, y0), (_f1, _x1, y1) in zip(track.samples[:-1], track.samples[1:]):
            total_steps += 1
            if y1 < y0:
                negative_steps += 1
        return total_steps > 0 and (negative_steps / total_steps) >= self.SHOT_MIN_NEGATIVE_STEP_RATIO

    def _select_shooter(self, track: BallTrack, robot_refs):
        if not robot_refs:
            return None, float("inf")
        first = track.first_point
        if first is None:
            return None, float("inf")
        _f0, x0, y0 = first
        origin_x, origin_y = x0, y0
        if len(track.samples) >= 2:
            deltas = []
            for (_fa, xa, ya), (_fb, xb, yb) in zip(track.samples[:-1], track.samples[1:]):
                deltas.append((xb - xa, yb - ya))
                if len(deltas) >= 3:
                    break
            if deltas:
                avg_dx = sum(dx for dx, _dy in deltas) / len(deltas)
                avg_dy = sum(dy for _dx, dy in deltas) / len(deltas)
                origin_x = x0 - 1.35 * avg_dx
                origin_y = y0 - 1.35 * avg_dy

        best_ref = None
        best_score = float("inf")
        for ref in robot_refs:
            rx, ry = ref["img"]
            dist_origin = math.hypot(origin_x - rx, origin_y - ry)
            dist_first = math.hypot(x0 - rx, y0 - ry)
            above_penalty = max(0.0, (y0 - ry) - 6.0) * 0.75
            score = 0.7 * dist_origin + 0.3 * dist_first + above_penalty
            if score < best_score:
                best_score = score
                best_ref = ref
        return best_ref, best_score

    def _moving_away_from_shooter(self, track: BallTrack) -> bool:
        if track.shooter_img is None or len(track.samples) < 2:
            return False
        first = track.first_point
        last = track.last_point
        if first is None or last is None:
            return False
        sx, sy = track.shooter_img
        _f0, x0, y0 = first
        _f1, x1, y1 = last
        start_dist = math.hypot(x0 - sx, y0 - sy)
        last_dist = math.hypot(x1 - sx, y1 - sy)
        return last_dist >= start_dist + self.SHOT_MIN_AWAY_FROM_SHOOTER_PX

    def _confirmed_goal_make(self, track: BallTrack) -> bool:
        if not track.entered_goal:
            return False
        if track.entered_goal_frames >= self.GOAL_CONFIRM_MIN_FRAMES:
            return True
        if track.first_goal_entry_frame is None:
            return False
        if track.missing_frames <= self.GOAL_ENTRY_MAX_MISSING and self._track_finished_inside_goal(track):
            return True
        return False

    def _track_finished_inside_goal(self, track: BallTrack) -> bool:
        if not track.goal_color or track.last_goal_entry_point is None:
            return False
        poly = self._goal_openings.get(track.goal_color)
        if poly is None:
            return False
        x, y = track.last_goal_entry_point
        return self.cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    def _is_duplicate_event(self, event: ShotEvent) -> bool:
        last_time = self._last_event_time_by_robot.get(event.shooter_id)
        if last_time is not None and (event.timestamp_s - last_time) < self.SHOT_DEDUP_TIME_S:
            return True
        self._last_event_time_by_robot[event.shooter_id] = event.timestamp_s
        return False

    def _track_reached_top(self, track: BallTrack) -> bool:
        if not track.samples:
            return False
        top_y = min(y for _f, _x, y in track.samples)
        goal_top = min(
            min(poly[:, 0, 1]) for poly in self._goal_approaches.values()
        ) if self._goal_approaches else 0.0
        goal_bottom = max(
            max(poly[:, 0, 1]) for poly in self._goal_approaches.values()
        ) if self._goal_approaches else 0.0
        return top_y <= goal_bottom + max(14.0, 0.2 * max(goal_bottom - goal_top, 1.0))

    def _detect_goal_openings(self, background, field_mask):
        cv2, np = self.cv2, self.np
        hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
        h, w = background.shape[:2]
        top_band = np.zeros((h, w), dtype=np.uint8)
        top_band[:max(1, int(round(h * 0.55))), :] = 255

        masks = {
            "blue": cv2.inRange(
                hsv,
                np.array(self.GOAL_BLUE_HSV_LO, dtype=np.uint8),
                np.array(self.GOAL_BLUE_HSV_HI, dtype=np.uint8),
            ),
            "red": cv2.bitwise_or(
                cv2.inRange(
                    hsv,
                    np.array(self.GOAL_RED1_HSV_LO, dtype=np.uint8),
                    np.array(self.GOAL_RED1_HSV_HI, dtype=np.uint8),
                ),
                cv2.inRange(
                    hsv,
                    np.array(self.GOAL_RED2_HSV_LO, dtype=np.uint8),
                    np.array(self.GOAL_RED2_HSV_HI, dtype=np.uint8),
                ),
            ),
        }

        openings = {}
        for color, mask in masks.items():
            mask = cv2.bitwise_and(mask, top_band)
            if color == "blue":
                mask[:, w // 2:] = 0
            else:
                mask[:, :w // 2] = 0
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            chosen = None
            for c in cnts:
                area = float(cv2.contourArea(c))
                if area < self.GOAL_MIN_AREA_PX:
                    continue
                x, y, ww, hh = cv2.boundingRect(c)
                if y > int(0.22 * h):
                    continue
                chosen = c
                break
            if chosen is None:
                continue
            x, y, ww, hh = cv2.boundingRect(chosen)
            pts = chosen.reshape(-1, 2)
            top_pts = pts[pts[:, 1] <= y + hh * self.GOAL_OPENING_TOP_FRAC]
            if len(top_pts) >= 3:
                hull = cv2.convexHull(top_pts.reshape(-1, 1, 2).astype(np.int32))
            else:
                hull = cv2.convexHull(chosen)
            openings[color] = self._shrink_polygon(
                hull,
                shrink_x=self.GOAL_OPENING_SHRINK_X,
                shrink_y=self.GOAL_OPENING_SHRINK_Y,
            )
        return openings

    def _fallback_goal_openings(self, tracker):
        if tracker._H_inv is None:
            return {}
        cv2, np = self.cv2, self.np
        field_points = {
            "blue": [(-8.0, -72.0), (-8.0, -40.0), (-36.0, -56.0)],
            "red": [(8.0, -72.0), (36.0, -56.0), (8.0, -40.0)],
        }
        openings = {}
        for color, pts_center in field_points.items():
            pts_corner = [_field_center_to_corner_xy(x, y) for x, y in pts_center]
            arr = np.array([pts_corner], dtype=np.float32)
            img = cv2.perspectiveTransform(arr, tracker._H_inv)[0]
            openings[color] = np.round(img).astype(np.int32).reshape(-1, 1, 2)
        return openings

    def _build_goal_approach(self, poly):
        pts = poly.reshape(-1, 2).astype(self.np.float32)
        x_min = float(self.np.min(pts[:, 0]))
        x_max = float(self.np.max(pts[:, 0]))
        y_min = float(self.np.min(pts[:, 1]))
        y_max = float(self.np.max(pts[:, 1]))
        w = max(x_max - x_min, 1.0)
        h = max(y_max - y_min, 1.0)
        pad_x = w * self.GOAL_APPROACH_SIDE_PAD_FRAC
        down = h * self.GOAL_APPROACH_DOWN_FRAC
        rect = self.np.array([
            [x_min - pad_x, y_min],
            [x_max + pad_x, y_min],
            [x_max + pad_x, y_max + down],
            [x_min - pad_x, y_max + down],
        ], dtype=self.np.int32)
        return rect.reshape(-1, 1, 2)

    def _shrink_polygon(self, poly, shrink_x: float, shrink_y: float):
        pts = poly.reshape(-1, 2).astype(self.np.float32)
        cx = float(self.np.mean(pts[:, 0]))
        cy = float(self.np.mean(pts[:, 1]))
        scaled = pts.copy()
        scaled[:, 0] = cx + (scaled[:, 0] - cx) * shrink_x
        scaled[:, 1] = cy + (scaled[:, 1] - cy) * shrink_y
        return self.np.round(scaled).astype(self.np.int32).reshape(-1, 1, 2)


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
    BALL_MIN_FIELD_AREA_IN2 = 2.0
    BALL_MASK_OPEN_FRAC = 0.030
    BALL_MASK_CLOSE_FRAC = 0.055
    BALL_MASK_DILATE_FRAC = 0.070
    INIT_MAX_CANDIDATES = 8
    INIT_MIN_SPACING_IN = 8.0
    REID_HIST_H_BINS = 18
    REID_HIST_S_BINS = 16
    REID_SAMPLE_STRIDE = 15
    REID_MAX_SAMPLES_PER_ROBOT = 48
    REID_CROP_SCALE = 0.55
    REID_COST_WEIGHT = 1.60
    REID_COST_WEIGHT_REACQ = 2.70
    POST_MERGE_LOCK_FRAMES = 12
    POST_MERGE_LOCK_WEIGHT = 1.25
    POST_MERGE_LOCK_REJECT_PX = 115.0
    POST_MERGE_LOCK_APPEAR_WEIGHT = 1.55
    POST_MERGE_DUPLICATE_PX = 32.0
    POST_MERGE_DUPLICATE_IN = 12.0

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
        self._track_features = [None] * 4

        self._merge_groups: Dict[FrozenSet, MergeGroup] = {}
        self._underresolved_tracks = set()
        self._post_merge_locks = [None] * 4
        self._last_ball_mask = None
        self._reid_refs: Optional[Dict[int, List]] = None
        # Static blob suppression: track per-blob pixel location history
        # Format: {approx_pixel_key: consecutive_static_frames}
        self._blob_static_counts: dict = {}
        self._blob_static_positions: dict = {}
        # How many frames a blob must stay within STATIC_BLOB_MOVE_PX to be suppressed
        self.STATIC_BLOB_SUPPRESS_FRAMES = 20
        # If blob moves less than this many pixels, count it as static
        self.STATIC_BLOB_MOVE_PX = 8

        # Corner structure exclusion zone (field coords, inches).
        # The physical corner posts create persistent FG blobs that trap R0/R3.
        # Suppress blobs inside these corner zones unless a track is actively moving.
        # Zone: x < CORNER_ZONE_IN  for y < CORNER_ZONE_IN  (top-left and top-right)
        self.CORNER_ZONE_IN = 22.0   # 22-inch square at each top corner
        self._frame_counter = 0      # for corner zone timing

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
            lock = self._post_merge_locks[i]
            if lock is not None:
                lock["frames_left"] -= 1
                if lock["frames_left"] <= 0:
                    self._post_merge_locks[i] = None
                else:
                    img_pos = lock["img_pos"]
                    img_vel = lock["img_vel"]
                    field_pos = lock["field_pos"]
                    field_vel = lock["field_vel"]
                    lock["img_pos"] = (img_pos[0] + img_vel[0], img_pos[1] + img_vel[1])
                    lock["field_pos"] = (field_pos[0] + field_vel[0], field_pos[1] + field_vel[1])

        blobs = self._get_blobs(frame)
        self._underresolved_tracks = self._find_underresolved_tracks(blobs)

        if not self._initialized:
            lineup = self._select_bootstrap_lineup(blobs)
            if lineup is None:
                for p in self.tracked_poses:
                    p.visible = False
                return self.tracked_poses
            self._initialize_tracks_from_lineup(lineup, "best visible 4-blob lineup")
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
                if len(mg.track_ids) > 2 and self._relabel_multi_merge_exit(mg, blobs, assignment):
                    for tid in mg.track_ids:
                        self._merge_recent[tid] = self.MERGE_HOLD
                else:
                    self._apply_separation(mg)

        self._prune_post_merge_duplicate_assignments(blobs, assignment)

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
                self._update_post_merge_lock(i, (fx, fy), (px, py))
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
                    feature = blobs[assignment[i]][9] if len(blobs[assignment[i]]) > 9 else None
                    self._track_features[i] = feature
                    self._update_post_merge_lock(i, (fx, fy), (cx, cy))
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

    def _update_post_merge_lock(self, track_id: int, field_pos, image_pos) -> None:
        lock = self._post_merge_locks[track_id]
        if lock is None:
            return
        prev_img = lock["img_pos"]
        prev_field = lock["field_pos"]
        lock["img_vel"] = (float(image_pos[0]) - prev_img[0], float(image_pos[1]) - prev_img[1])
        lock["field_vel"] = (float(field_pos[0]) - prev_field[0], float(field_pos[1]) - prev_field[1])
        lock["img_pos"] = (float(image_pos[0]), float(image_pos[1]))
        lock["field_pos"] = (float(field_pos[0]), float(field_pos[1]))

    def _prime_post_merge_lock(self, track_id: int, field_pos, image_pos,
                               prev_image_pos=None, feature=None,
                               merge_group_key=None) -> None:
        prev_field = self._pos[track_id] if self._pos[track_id] is not None else field_pos
        if prev_image_pos is None:
            img_vel = (0.0, 0.0)
        else:
            img_vel = (
                float(image_pos[0]) - float(prev_image_pos[0]),
                float(image_pos[1]) - float(prev_image_pos[1]),
            )
        field_vel = (
            float(field_pos[0]) - float(prev_field[0]),
            float(field_pos[1]) - float(prev_field[1]),
        )
        self._post_merge_locks[track_id] = {
            "frames_left": self.POST_MERGE_LOCK_FRAMES,
            "img_pos": (float(image_pos[0]), float(image_pos[1])),
            "img_vel": img_vel,
            "field_pos": (float(field_pos[0]), float(field_pos[1])),
            "field_vel": field_vel,
            "feature": feature if feature is not None else self._track_features[track_id],
            "merge_group_key": merge_group_key,
        }

    def _assignment_support_cost(self, track_id: int, blob) -> float:
        pred_field = self._predict_field_pos(track_id)
        pred_px_x, pred_px_y = self._predict_image_pos(track_id)
        d_field = math.hypot(float(blob[0]) - pred_field[0], float(blob[1]) - pred_field[1])
        d_img = math.hypot(float(blob[4]) - pred_px_x, float(blob[5]) - pred_px_y)
        cost = (
            d_field / max(self.MAX_REACQ_IN, 1.0)
            + 0.35 * d_img / max(self._robot_max_px * 1.25, 1.0)
            + float(blob[6])
            + self._appearance_cost(track_id, blob)
        )
        lock = self._post_merge_locks[track_id]
        if lock is not None:
            lock_img_x, lock_img_y = lock["img_pos"]
            d_lock_img = math.hypot(float(blob[4]) - lock_img_x, float(blob[5]) - lock_img_y)
            cost += self.POST_MERGE_LOCK_WEIGHT * (
                d_lock_img / max(self._robot_max_px, 1.0))
            cost += self.POST_MERGE_LOCK_APPEAR_WEIGHT * self._feature_distance(
                lock.get("feature"),
                blob[9] if len(blob) > 9 else None,
            )
        return cost

    def _prune_post_merge_duplicate_assignments(self, blobs, assignment: Dict[int, int]) -> None:
        if len(assignment) < 2:
            return

        removed = set()
        assigned_tracks = list(assignment.keys())
        for idx, tid_a in enumerate(assigned_tracks):
            if tid_a in removed or tid_a not in assignment:
                continue
            lock_a = self._post_merge_locks[tid_a]
            if lock_a is None or lock_a.get("merge_group_key") is None:
                continue
            blob_a = blobs[assignment[tid_a]]
            for tid_b in assigned_tracks[idx + 1:]:
                if tid_b in removed or tid_b not in assignment:
                    continue
                lock_b = self._post_merge_locks[tid_b]
                if lock_b is None or lock_b.get("merge_group_key") != lock_a.get("merge_group_key"):
                    continue
                blob_b = blobs[assignment[tid_b]]
                d_img = math.hypot(float(blob_a[4]) - float(blob_b[4]),
                                   float(blob_a[5]) - float(blob_b[5]))
                d_field = math.hypot(float(blob_a[0]) - float(blob_b[0]),
                                     float(blob_a[1]) - float(blob_b[1]))
                if d_img > self.POST_MERGE_DUPLICATE_PX or d_field > self.POST_MERGE_DUPLICATE_IN:
                    continue

                cost_a = self._assignment_support_cost(tid_a, blob_a)
                cost_b = self._assignment_support_cost(tid_b, blob_b)
                drop_tid = tid_a if cost_a > cost_b else tid_b
                removed.add(drop_tid)

        for tid in removed:
            del assignment[tid]
            self._post_merge_locks[tid] = None
            self._coast[tid] = self.VISIBLE_COAST
            self.tracked_poses[tid].visible = False
            print("[INFO] Pruned duplicate post-merge track: R{}".format(tid))

    def _initialize_tracks_from_lineup(self, lineup, reason: str):
        for i, (fx, fy, cx, cy) in enumerate(lineup):
            self._pos[i] = (fx, fy)
            self._pos_px[i] = (cx, cy)
            self._vel[i] = (0.0, 0.0)
            self._vel_px[i] = (0.0, 0.0)
            self._coast[i] = 0
            self.tracked_poses[i].x_in = fx
            self.tracked_poses[i].y_in = fy
            self.tracked_poses[i].heading = 0.0
            self.tracked_poses[i].visible = True
        self._initialized = True
        print("[INFO] Tracker initialized from {}.".format(reason))

    def _select_bootstrap_lineup(self, blobs):
        if len(blobs) < 4:
            return None

        candidate_pool = blobs[:min(len(blobs), self.INIT_MAX_CANDIDATES)]
        best = None
        best_score = float("inf")
        for combo in itertools.combinations(candidate_pool, 4):
            lineup = sorted(combo, key=lambda b: b[0])
            spacing_penalty = 0.0
            for left, right in zip(lineup, lineup[1:]):
                gap = right[0] - left[0]
                if gap < self.INIT_MIN_SPACING_IN:
                    spacing_penalty += (self.INIT_MIN_SPACING_IN - gap) ** 2

            score = sum(b[6] for b in lineup) + 0.04 * spacing_penalty
            if score < best_score:
                best_score = score
                best = [(float(b[0]), float(b[1]), float(b[4]), float(b[5]))
                        for b in lineup]
        return best

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
                lock = self._post_merge_locks[i]
                if lock is not None:
                    lock_img_x, lock_img_y = lock["img_pos"]
                    d_lock_img = math.hypot(b[4] - lock_img_x, b[5] - lock_img_y)
                    if d_lock_img > self.POST_MERGE_LOCK_REJECT_PX:
                        continue
                    lock_feature_cost = self._feature_distance(lock.get("feature"), b[9] if len(b) > 9 else None)
                else:
                    d_lock_img = 0.0
                    lock_feature_cost = 0.0
                dist_scale = (self.MAX_REACQ_IN
                              if self._coast[i] > self.VISIBLE_COAST
                              else self.MAX_DIST_IN)
                dist_cost = d_field / max(dist_scale, 1.0)
                img_cost  = d_img / max(self._robot_max_px * 1.25, 1.0)
                qual_cost = b[6]
                appearance_cost = self._appearance_cost(i, b)
                appearance_weight = (
                    self.REID_COST_WEIGHT_REACQ
                    if self._coast[i] > self.VISIBLE_COAST
                    else self.REID_COST_WEIGHT
                )
                real_cost[i, j] = (
                    dist_cost
                    + 0.35 * img_cost
                    + qual_cost
                    + appearance_weight * appearance_cost
                    + self.POST_MERGE_LOCK_WEIGHT * (
                        d_lock_img / max(self._robot_max_px, 1.0))
                    + self.POST_MERGE_LOCK_APPEAR_WEIGHT * lock_feature_cost
                )

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
                if self._post_merge_locks[i] is not None:
                    skip_cost_i += 0.6
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
        order_votes = {
            tid: [0.0] * len(entry_order)
            for tid in track_ids
        }
        entry_features = {
            tid: self._track_features[tid]
            for tid in track_ids
            if self._track_features[tid] is not None
        }
        for slot, tid in enumerate(entry_order):
            order_votes[tid][slot] += 1.0

        return MergeGroup(
            track_ids      = list(track_ids),
            entry_axis     = (ax, ay),
            parent_id      = parent_id,
            crossed        = False,
            entry_order    = entry_order,
            current_order  = list(entry_order),   # starts as identity permutation
            peak_assignment= peak_assignment,
            order_votes    = order_votes,
            entry_features = entry_features,
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

        # Sort tracks by current projection so nearby tracks compete fairly
        track_current_px.sort(key=lambda t: _proj(t[1], t[2]))
        successfully_assigned = 0
        used_peaks = set()
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

        projected_order = sorted(
            mg.track_ids,
            key=lambda tid: _proj(*mg.peak_assignment[tid])
            if tid in mg.peak_assignment else 0.0
        )
        mg.current_order = projected_order
        self._record_merge_order(
            mg,
            projected_order,
            weight=1.0 if successfully_assigned >= n else 0.35,
        )

    def _record_merge_order(self, mg: MergeGroup, order: List[int], weight: float) -> None:
        if not mg.order_votes or not order:
            return
        for slot, tid in enumerate(order):
            if tid not in mg.order_votes:
                mg.order_votes[tid] = [0.0] * len(order)
            if slot < len(mg.order_votes[tid]):
                mg.order_votes[tid][slot] += weight

    def _resolve_merge_order(self, mg: MergeGroup) -> List[int]:
        if not mg.order_votes:
            return list(mg.current_order or mg.entry_order)

        best_order = None
        best_score = float("-inf")
        for order in itertools.permutations(mg.track_ids):
            score = 0.0
            for slot, tid in enumerate(order):
                score += mg.order_votes.get(tid, [0.0] * len(order))[slot]
            if score > best_score:
                best_score = score
                best_order = list(order)
        return best_order or list(mg.current_order or mg.entry_order)

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

    def _relabel_multi_merge_exit(self, mg: MergeGroup, blobs, assignment: Dict[int, int]) -> bool:
        if not blobs or len(mg.track_ids) <= 2:
            return False

        expected_order = self._resolve_merge_order(mg)
        if len(expected_order) != len(mg.track_ids):
            expected_order = list(mg.current_order or mg.entry_order)
        if len(expected_order) != len(mg.track_ids):
            return False

        slot_for_tid = {tid: slot for slot, tid in enumerate(expected_order)}
        ax, ay = mg.entry_axis
        peak_points = [mg.peak_assignment.get(tid) for tid in mg.track_ids if tid in mg.peak_assignment]
        if not peak_points:
            return False
        center_x = sum(px for px, _py in peak_points) / len(peak_points)
        center_y = sum(py for _px, py in peak_points) / len(peak_points)

        def _proj(px, py):
            return (px - center_x) * ax + (py - center_y) * ay

        candidate_set = set()
        for tid in mg.track_ids:
            if tid in assignment:
                candidate_set.add(assignment[tid])

        blob_ranked = []
        for bi, blob in enumerate(blobs):
            cx, cy = float(blob[4]), float(blob[5])
            nearest_peak = min(
                math.hypot(cx - px, cy - py)
                for px, py in peak_points
            )
            if nearest_peak <= self._robot_max_px * 2.6:
                blob_ranked.append((nearest_peak, bi))

        blob_ranked.sort()
        for _dist, bi in blob_ranked:
            candidate_set.add(bi)
            if len(candidate_set) >= min(len(blobs), len(mg.track_ids) + 4):
                break

        # Pull in the top continuation candidates for each track separately.
        # This matters when a robot explodes outward on the split frame and is
        # slightly farther from the merge peaks than the tighter cluster blobs.
        for tid in mg.track_ids:
            peak = mg.peak_assignment.get(tid)
            pred_field = self._predict_field_pos(tid)
            entry_feature = mg.entry_features.get(tid, self._track_features[tid])
            per_track = []
            for bi, blob in enumerate(blobs):
                d_peak = 0.0
                if peak is not None:
                    d_peak = math.hypot(float(blob[4]) - peak[0], float(blob[5]) - peak[1])
                d_field = math.hypot(float(blob[0]) - pred_field[0], float(blob[1]) - pred_field[1])
                appearance_cost = self._appearance_cost(tid, blob)
                entry_feature_cost = self._feature_distance(entry_feature, blob[9] if len(blob) > 9 else None)
                per_track.append((
                    d_peak / max(self._robot_max_px, 1.0)
                    + 0.6 * d_field / max(self.MAX_REACQ_IN, 1.0)
                    + 1.4 * appearance_cost
                    + 1.1 * entry_feature_cost,
                    bi,
                ))
            per_track.sort()
            for _score, bi in per_track[:3]:
                candidate_set.add(bi)

        candidate_indices = list(candidate_set)
        if len(candidate_indices) < len(mg.track_ids):
            return False

        sorted_candidates = sorted(candidate_indices, key=lambda bi: _proj(blobs[bi][4], blobs[bi][5]))

        best_cost = float("inf")
        best_map = None
        for chosen in itertools.combinations(sorted_candidates, len(mg.track_ids)):
            total_cost = 0.0
            valid = True
            for slot, (tid, bi) in enumerate(zip(expected_order, chosen)):
                blob = blobs[bi]
                peak = mg.peak_assignment.get(tid)
                if peak is None:
                    peak = self._predict_image_pos(tid)
                pred_field = self._predict_field_pos(tid)
                d_img = math.hypot(float(blob[4]) - peak[0], float(blob[5]) - peak[1])
                d_field = math.hypot(float(blob[0]) - pred_field[0], float(blob[1]) - pred_field[1])
                appearance_cost = self._appearance_cost(tid, blob)
                entry_feature = mg.entry_features.get(tid, self._track_features[tid])
                entry_feature_cost = self._feature_distance(
                    entry_feature,
                    blob[9] if len(blob) > 9 else None,
                )
                slot_penalty = abs(slot - slot_for_tid[tid])
                lock = self._post_merge_locks[tid]
                if lock is not None:
                    lock_img_x, lock_img_y = lock["img_pos"]
                    d_lock = math.hypot(float(blob[4]) - lock_img_x, float(blob[5]) - lock_img_y)
                else:
                    d_lock = 0.0

                total_cost += (
                    d_img / max(self._robot_max_px, 1.0)
                    + 0.55 * d_field / max(self.MAX_DIST_IN, 1.0)
                    + 1.9 * appearance_cost
                    + 1.35 * entry_feature_cost
                    + 0.65 * slot_penalty
                    + 0.8 * d_lock / max(self._robot_max_px, 1.0)
                )

                if d_img > self._robot_max_px * 3.6 and appearance_cost > 0.25:
                    valid = False
                    break

            if valid and total_cost < best_cost:
                best_cost = total_cost
                best_map = {
                    tid: bi
                    for tid, bi in zip(expected_order, chosen)
                }

        if best_map is None:
            return False

        chosen_blob_indices = set(best_map.values())
        for tid, bi in list(assignment.items()):
            if tid not in mg.track_ids and bi in chosen_blob_indices:
                del assignment[tid]
        for tid, bi in best_map.items():
            assignment[tid] = bi
            fx, fy = float(blobs[bi][0]), float(blobs[bi][1])
            cx, cy = float(blobs[bi][4]), float(blobs[bi][5])
            feature = blobs[bi][9] if len(blobs[bi]) > 9 else None
            self._track_features[tid] = feature
            self._prime_post_merge_lock(
                tid,
                (fx, fy),
                (cx, cy),
                prev_image_pos=mg.peak_assignment.get(tid),
                feature=feature,
                merge_group_key=tuple(sorted(mg.track_ids)),
            )

        swaps = []
        for slot, tid in enumerate(expected_order):
            previous_tid = mg.entry_order[slot] if slot < len(mg.entry_order) else tid
            if previous_tid != tid:
                swaps.append("R{}←slot{}".format(previous_tid, slot))
        print("[INFO] Multi-robot merge relabeled from exit blobs: {}".format(
            sorted(mg.track_ids)))
        return True

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
        self._frame_counter += 1
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
                # ── Corner zone suppression ─────────────────────────────────
                # Blobs inside the top-left or top-right corner zones are
                # suppressed unless a robot is known to be actively moving there.
                cz = self.CORNER_ZONE_IN
                in_corner_zone = (
                    (fx < cz and fy < cz) or           # top-left corner
                    (fx > 144 - cz and fy < cz)        # top-right corner
                )
                if self._initialized and in_corner_zone:
                    # Allow if an actively tracked robot is already nearby.
                    robot_present = False
                    for tid in range(4):
                        if self._pos[tid] is None:
                            continue
                        tx, ty = self._pos[tid]
                        dist = math.hypot(fx - tx, fy - ty)
                        if dist < self.ROBOT_SIZE_IN * 1.5 and self._coast[tid] <= self.VISIBLE_COAST:
                            robot_present = True
                            break
                    if not robot_present:
                        continue
                # ── end corner zone suppression ─────────────────────────────
                appearance = self._extract_appearance_feature(frame, int(cx), int(cy), c)
                candidates.append((fx, fy, 0.0, per_area, parent_id,
                                    int(cx), int(cy), quality_penalty,
                                    len(sub_centers), c, appearance))

        candidates.sort(key=lambda b: -b[3])

        # ── Static blob suppression ─────────────────────────────────────────
        # Corner structures and other stationary FG objects create persistent
        # blobs that confuse robot tracking.  Any candidate whose centroid has
        # stayed within STATIC_BLOB_MOVE_PX pixels for STATIC_BLOB_SUPPRESS_FRAMES
        # consecutive frames is dropped UNLESS a current track is already assigned
        # very close to it (which would mean it really is a robot sitting still).
        if self._initialized:
            filtered = []
            new_static: dict = {}
            new_static_positions: dict = {}
            for entry in candidates:
                fx, fy, hdg, _a, pid, cx, cy, qual, nsplit, cnt, appearance = entry
                # Bucket this centroid to a coarse grid for history lookup
                key = (round(cx / self.STATIC_BLOB_MOVE_PX),
                       round(cy / self.STATIC_BLOB_MOVE_PX))
                # Find closest existing static key
                matched_key = None
                for k in self._blob_static_positions:
                    okx, oky = self._blob_static_positions[k]
                    if abs(cx - okx) < self.STATIC_BLOB_MOVE_PX and abs(cy - oky) < self.STATIC_BLOB_MOVE_PX:
                        matched_key = k
                        break
                if matched_key is not None:
                    count = self._blob_static_counts.get(matched_key, 0) + 1
                    new_static[matched_key] = count
                    new_static_positions[matched_key] = (cx, cy)
                    near_track = any(
                        self._pos[tid] is not None
                        and math.hypot(fx - self._pos[tid][0], fy - self._pos[tid][1]) < self.ROBOT_SIZE_IN * 1.5
                        and self._coast[tid] <= self.VISIBLE_COAST
                        for tid in range(4)
                    )
                    if count >= self.STATIC_BLOB_SUPPRESS_FRAMES and not near_track:
                        continue
                else:
                    new_static[key] = 1
                    new_static_positions[key] = (cx, cy)
                filtered.append(entry)
            self._blob_static_counts = new_static
            self._blob_static_positions = new_static_positions
            candidates = filtered
        else:
            self._blob_static_counts = {}
            self._blob_static_positions = {}
        # ── end static blob suppression ─────────────────────────────────────

        return [(fx, fy, hdg, pid, cx, cy, qual, nsplit, cnt, appearance)
                for fx, fy, hdg, _a, pid, cx, cy, qual, nsplit, cnt, appearance in candidates]

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

    def _extract_appearance_feature(self, frame, cx: int, cy: int, contour=None):
        cv2, np = self.cv2, self.np
        if frame is None:
            return None

        radius = max(6, int(round(self._robot_max_px * self.REID_CROP_SCALE)))
        x0 = max(0, int(cx) - radius)
        y0 = max(0, int(cy) - radius)
        x1 = min(frame.shape[1], int(cx) + radius + 1)
        y1 = min(frame.shape[0], int(cy) + radius + 1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None

        crop = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.circle(mask, (int(cx) - x0, int(cy) - y0), max(3, radius - 1), 255, -1)
        if contour is not None:
            shifted = contour.copy()
            shifted[:, 0, 0] -= x0
            shifted[:, 0, 1] -= y0
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [shifted], -1, 255, -1)
            mask = cv2.bitwise_and(mask, contour_mask)

        hist = cv2.calcHist(
            [hsv], [0, 1], mask,
            [self.REID_HIST_H_BINS, self.REID_HIST_S_BINS],
            [0, 180, 0, 256],
        )
        if hist is None:
            return None
        hist = hist.astype(np.float32)
        total = float(hist.sum())
        if total <= 0.0:
            return None
        hist /= total
        return hist

    def _appearance_cost(self, track_id: int, blob) -> float:
        if len(blob) < 10 or blob[9] is None:
            return 0.0
        return self._appearance_cost_for_feature(track_id, blob[9])

    def _feature_distance(self, feature_a, feature_b) -> float:
        if feature_a is None or feature_b is None:
            return 0.0
        return float(self.cv2.compareHist(feature_a, feature_b, self.cv2.HISTCMP_BHATTACHARYYA))

    def _appearance_cost_for_feature(self, track_id: int, feature) -> float:
        if not self._reid_refs or track_id not in self._reid_refs or feature is None:
            return 0.0

        refs = self._reid_refs.get(track_id) or []
        if not refs:
            return 0.0

        cv2 = self.cv2
        best = 1.0
        for ref in refs:
            d = float(cv2.compareHist(feature, ref, cv2.HISTCMP_BHATTACHARYYA))
            if d < best:
                best = d
        return best

    def set_reid_reference_histograms(self, refs: Dict[int, List]) -> None:
        self._reid_refs = refs if refs else None


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


def _load_manual_pose_rows(path: str):
    if sniff_jlog(path):
        return read_jlog_rows(path)
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _build_manual_reid_histograms(
    tracker,
    cap,
    manual_reference_csv: str,
    H_inv,
    video_fps: float,
):
    cv2, np = tracker.cv2, tracker.np
    rows = _load_manual_pose_rows(manual_reference_csv)
    if not rows:
        print("[WARN] Manual re-ID reference file is empty: {}".format(manual_reference_csv))
        return None

    refs = {i: [] for i in range(4)}
    max_samples = tracker.REID_MAX_SAMPLES_PER_ROBOT
    stride = max(1, tracker.REID_SAMPLE_STRIDE)

    for row_idx in range(0, len(rows), stride):
        row = rows[row_idx]
        frame_num = int(round(float(row["timestamp_s"]) * video_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        for robot_id in range(4):
            if row["robot{}_visible".format(robot_id)] != "1":
                continue
            if len(refs[robot_id]) >= max_samples:
                continue

            fx_center = float(row["robot{}_x_in".format(robot_id)])
            fy_center = float(row["robot{}_y_in".format(robot_id)])
            fx, fy = _field_center_to_corner_xy(fx_center, fy_center)
            pt = np.array([[[fx, fy]]], dtype=np.float32)
            ip = cv2.perspectiveTransform(pt, H_inv)[0][0]
            hist = tracker._extract_appearance_feature(
                frame, int(round(float(ip[0]))), int(round(float(ip[1]))))
            if hist is not None:
                refs[robot_id].append(hist)

        if all(len(refs[i]) >= max_samples for i in range(4)):
            break

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    counts = [len(refs[i]) for i in range(4)]
    if not any(counts):
        print("[WARN] Could not build manual re-ID appearance references.")
        return None

    print("[INFO] Manual re-ID samples per robot: {}".format(counts))
    return refs


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


def _open_debug_video_writer(cv2, base_output_dir, frame_size, fps):
    candidates = [
        ("tracker_debug.mp4", "mp4v"),
        ("tracker_debug.mp4", "avc1"),
        ("tracker_debug.avi", "MJPG"),
        ("tracker_debug.avi", "XVID"),
    ]
    width, height = frame_size
    attempted = []
    for filename, codec in candidates:
        path = os.path.join(base_output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer, path, codec
        attempted.append("{} ({})".format(path, codec))
        writer.release()
    return None, attempted, None


# ─────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ─────────────────────────────────────────────────────────────────────────────
def process_match(
    video_path,
    output_dir,
    start_offset_sec  = 0.0,
    sample_rate_fps   = 10.0,
    debug             = False,
    debug_video       = False,
    debug_every_n     = 1,
    debug_enable_hitboxes = False,
    manual_corners_px = None,
    robot_init_positions = None,  # list of 4 [x_in, y_in] center-origin field coords sorted by robot id
    manual_reference_csv = None,
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
        if debug_video:
            print("[INFO] Writing debug video from frames sampled about every {} source frames".format(
                debug_every_n))
        else:
            print("[INFO] Saving debug frame about every {} source frames".format(debug_every_n))
        if debug_enable_hitboxes:
            print("[INFO] Debug robot wireframe hitboxes enabled.")

    csv_path    = os.path.join(output_dir, "robot_positions.csv")
    jlog_path   = os.path.join(output_dir, "robot_positions.jlog")
    wpilog_path = os.path.join(output_dir, "match_log.wpilog")
    debug_dir   = os.path.join(output_dir, "tracker_debug") if (debug and not debug_video) else None
    debug_video_path = os.path.join(output_dir, "tracker_debug.mp4") if (debug and debug_video) else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        for name in os.listdir(debug_dir):
            if name.endswith(".jpg"):
                try:
                    os.remove(os.path.join(debug_dir, name))
                except OSError:
                    pass
    if debug_video_path and os.path.exists(debug_video_path):
        try:
            os.remove(debug_video_path)
        except OSError:
            pass
    avi_fallback_path = os.path.join(output_dir, "tracker_debug.avi")
    if debug_video_path and os.path.exists(avi_fallback_path):
        try:
            os.remove(avi_fallback_path)
        except OSError:
            pass

    log = WPILogWriter(wpilog_path)
    pose_eids = [log.start_entry("Robot{}/Pose".format(i),    "double[]") for i in range(4)]
    vis_eids  = [log.start_entry("Robot{}/Visible".format(i), "boolean")  for i in range(4)]

    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_COLUMNS)
    jlog_writer = JuiceLogWriter(jlog_path, schema=ROBOT_POSE_SCHEMA)

    tracker        = RobotTracker(cv2, np)
    shot_detector  = ShotDetector(cv2, np)
    field_detector = FieldDetector(cv2, np)

    ordered = None; H_2d = None; H_inv = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, sample_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame."); sys.exit(1)
    frame_shape = sample_frame.shape
    debug_writer = None
    if debug_video_path:
        debug_h, debug_w = frame_shape[:2]
        debug_fps = max(1.0, video_fps / max(1, debug_every_n))
        debug_writer, opened_path_or_attempts, debug_codec = _open_debug_video_writer(
            cv2,
            output_dir,
            (debug_w, debug_h),
            debug_fps,
        )
        if debug_writer is None:
            print("[ERROR] Could not open debug video for writing. Tried:")
            for attempt in opened_path_or_attempts:
                print("  - {}".format(attempt))
            sys.exit(1)
        debug_video_path = opened_path_or_attempts
        print("[INFO] Debug video: {} ({:.2f} fps, codec {})".format(
            debug_video_path, debug_fps, debug_codec))

    if manual_corners_px is not None:
        bl, br, tr, tl = [np.array(c, np.float32) for c in manual_corners_px]
        ordered = np.array([tl, tr, br, bl], np.float32)
        H_2d, _ = cv2.findHomography(
            ordered, np.array([[0,0],[144,0],[144,144],[0,144]], np.float32))
        H_inv = np.linalg.inv(H_2d)
        tracker.setup(video_path, ordered, frame_shape)
        shot_detector.setup(tracker)
        print("[INFO] Manual corners loaded.")
        if tracker._bg is not None:
            bg_path = os.path.join(output_dir, "median_background.jpg")
            cv2.imwrite(bg_path, tracker._bg)
            print("[INFO] Median background saved: {}".format(bg_path))
    else:
        print("[WARN] No corners — will attempt auto-detection.")

    # --- Manual robot initialization (bypasses blob-based init) ---
    if robot_init_positions is not None and H_inv is not None:
        print("[INFO] Using manual robot init positions — bypassing blob-based init.")
        for i, (fx_center, fy_center) in enumerate(robot_init_positions[:4]):
            fx, fy = _field_center_to_corner_xy(fx_center, fy_center)
            pt = np.array([[[float(fx), float(fy)]]], np.float32)
            ip = cv2.perspectiveTransform(pt, H_inv)[0][0]
            cx, cy = float(ip[0]), float(ip[1])
            tracker._pos[i]    = (fx, fy)
            tracker._pos_px[i] = (cx, cy)
            tracker._coast[i]  = 0
            tracker.tracked_poses[i].x_in    = fx
            tracker.tracked_poses[i].y_in    = fy
            tracker.tracked_poses[i].heading = 0.0
            tracker.tracked_poses[i].visible = True
            print("[INFO]   Robot{}: field_center=({:.1f},{:.1f}) pixel=({:.0f},{:.0f})".format(
                i, fx_center, fy_center, cx, cy))
        tracker._initialized = True
        print("[INFO] Tracker force-initialized with {} robots.".format(
            len(robot_init_positions[:4])))

    if manual_reference_csv is not None and H_inv is not None:
        refs = _build_manual_reid_histograms(
            tracker=tracker,
            cap=cap,
            manual_reference_csv=manual_reference_csv,
            H_inv=H_inv,
            video_fps=video_fps,
        )
        if refs:
            tracker.set_reid_reference_histograms(refs)
            print("[INFO] Manual re-ID appearance model enabled.")

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
                shot_detector.setup(tracker)
                print("\n  [t={:.1f}s] Field auto-detected.".format(match_time_s))
                if tracker._bg is not None:
                    cv2.imwrite(os.path.join(output_dir, "median_background.jpg"),
                                tracker._bg)

        poses = tracker.update(frame)
        shot_events = shot_detector.update(current_frame_num, match_time_s, poses, tracker)
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
            log.write_pose2d(
                pose_eids[i],
                timestamp_us,
                p.wpilog_x_m,
                p.wpilog_y_m,
                p.wpilog_heading_rad,
            )
            log.write_boolean(vis_eids[i], timestamp_us, p.visible)

        row = {"timestamp_s": "{:.4f}".format(match_time_s)}
        for robot_id, p in enumerate(poses):
            row["robot{}_x_in".format(robot_id)] = "{:.2f}".format(p.x_center_in)
            row["robot{}_y_in".format(robot_id)] = "{:.2f}".format(p.y_center_in)
            row["robot{}_heading_rad".format(robot_id)] = "{:.4f}".format(p.heading)
            row["robot{}_visible".format(robot_id)] = "1" if p.visible else "0"
        shot_cells = [["", "", "", ""] for _ in range(4)]
        for event in shot_events:
            if 0 <= event.shooter_id < 4:
                shot_cells[event.shooter_id] = [
                    event.result,
                    "{:.2f}".format(event.shot_x_in),
                    "{:.2f}".format(event.shot_y_in),
                    event.goal_color,
                ]
        for robot_id, cells in enumerate(shot_cells):
            row["robot{}_shot_result".format(robot_id)] = cells[0]
            row["robot{}_shot_x_in".format(robot_id)] = cells[1]
            row["robot{}_shot_y_in".format(robot_id)] = cells[2]
            row["robot{}_shot_goal".format(robot_id)] = cells[3]
        csv_writer.writerow(csv_row_to_list(row))
        jlog_writer.append_row(row)

        if debug and H_inv is not None and (debug_dir or debug_writer is not None):
            dbg = frame.copy()
            fg_debug = None
            if ordered is not None:
                cv2.polylines(dbg, [ordered.astype(np.int32).reshape(-1,1,2)],
                              True, (0, 200, 0), 2)
                shot_detector.draw_debug(dbg)

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
                if debug_writer is not None:
                    debug_writer.write(dbg)
                elif debug_dir is not None:
                    cv2.imwrite(os.path.join(debug_dir,
                                "frame_{:06d}.jpg".format(current_frame_num)), dbg)
                if current_frame_num >= next_debug_source_frame:
                    while current_frame_num >= next_debug_source_frame:
                        next_debug_source_frame += base_debug_interval

        for event in shot_events:
            print("[INFO] Shot {} by R{} at ({:.1f},{:.1f}) goal={}".format(
                event.result,
                event.shooter_id,
                event.shot_x_in,
                event.shot_y_in,
                event.goal_color or "unknown",
            ))

        processed += 1

    bar.finish()
    if debug_writer is not None:
        debug_writer.release()
    log.close(); csv_file.close(); jlog_writer.close(); cap.release()
    print("\n[DONE] {} frames processed.".format(processed))
    print("  CSV:    {}".format(csv_path))
    print("  JLOG:   {}".format(jlog_path))
    print("  WPILOG: {}".format(wpilog_path))
    if debug:
        print("  Debug:  {}".format(debug_video_path if debug_video_path else debug_dir))
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
    p.add_argument("--debug-video",  action="store_true",
                   help="Write annotated debug output to tracker_debug.mp4 instead of tracker_debug/ images; implies --debug")
    p.add_argument("--debug-every",  type=int, default=1,
                   help="Save 1 debug frame every N processed frames (default 1)")
    p.add_argument("--debug-enable-hitboxes", action="store_true",
                   help="Draw robot wireframe hitboxes in debug frames")
    p.add_argument("--no-download",  action="store_true")
    p.add_argument("--video-path",   default=None)
    p.add_argument("--corners",      default=None,
                   help="field_corners.json from ftc_calibrate.py")
    p.add_argument("--robot-init-positions", default=None,
                   help=(
                       "JSON file or inline JSON string with starting field positions "
                       "for all 4 robots, sorted by robot id. "
                       "Format: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] in inches "
                       "with (0,0) at field center. "
                       "Bypasses blob-based init (required when robots start under "
                       "corner structures or are otherwise hard to detect at t=0). "
                       "Example: --robot-init-positions "
                       "'[[-52.7,-70.4],[-12.6,53.0],[15.8,60.6],[57.5,-55.7]]'"))
    p.add_argument("--manual-reference-csv", default=None,
                   help=(
                       "Manual robot_positions-style CSV or JLOG used to build a supervised "
                       "appearance re-ID model for this video. "
                       "Coordinates are interpreted with (0,0) at field center. "
                       "Useful for tuning tracker identity assignment against "
                       "hand-labeled data."
                   ))
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

    robot_init = None
    if args.robot_init_positions:
        raw = args.robot_init_positions.strip()
        if os.path.isfile(raw):
            with open(raw) as f:
                robot_init = json.load(f)
        else:
            robot_init = json.loads(raw)
        print("[INFO] Robot init positions:", robot_init)

    process_match(
        video_path           = video_path,
        output_dir           = args.output_dir,
        start_offset_sec     = args.start_offset,
        sample_rate_fps      = args.sample_rate,
        debug                = (args.debug or args.debug_video),
        debug_video          = args.debug_video,
        debug_every_n        = args.debug_every,
        debug_enable_hitboxes= args.debug_enable_hitboxes,
        manual_corners_px    = corners,
        robot_init_positions = robot_init,
        manual_reference_csv = args.manual_reference_csv,
    )


if __name__ == "__main__":
    main()
