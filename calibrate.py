# -*- coding: utf-8 -*-
"""
ftc_calibrate.py — Interactive field corner picker for FTC match videos.

Usage:
    python3 ftc_calibrate.py <video_path> [--output field_corners.json] [--frame N]

Controls:
    - Drag any of the 4 corner points to reposition them
    - Scroll wheel or +/- keys to zoom in/out
    - Middle-click drag (or right-click drag) to pan
    - R         — reset corners to default positions
    - ENTER / S — save and exit
    - ESC / Q   — quit without saving

The 4 corners are color-coded:
    TL = yellow  (top-left  of field)
    TR = cyan    (top-right of field)
    BR = red     (bottom-right of field)
    BL = blue    (bottom-left of field)

Output JSON format (compatible with auto_scout.py --corners):
    {"corners_px": [[bl_x, bl_y], [br_x, br_y], [tr_x, tr_y], [tl_x, tl_y]]}
"""

import argparse
import json
import math
import sys

def _require(package, pip_name=None):
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        name = pip_name or package
        print(f"[ERROR] Missing dependency: {name}\n  Install with:  pip install {name}")
        sys.exit(1)


def get_middle_frame(video_path, frame_number=None):
    cv2 = _require("cv2", "opencv-python")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if frame_number is None:
        frame_number = total // 2
    frame_number = max(0, min(frame_number, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Could not read frame from video.")
        sys.exit(1)
    t = frame_number / fps
    print(f"[INFO] Loaded frame {frame_number}/{total} (t={t:.1f}s) — {frame.shape[1]}x{frame.shape[0]}")
    return frame


def default_corners(w, h):
    """Default corners placed at 20%/80% of frame dimensions."""
    mx, my = int(w * 0.20), int(h * 0.20)
    px, py = int(w * 0.80), int(h * 0.80)
    # Order: TL, TR, BR, BL
    return [
        [mx, my],
        [px, my],
        [px, py],
        [mx, py],
    ]


LABELS = ["TL", "TR", "BR", "BL"]
# BGR colors
COLORS = [
    (0,   220, 220),  # TL — yellow-ish (actually cyan-yellow)
    (220, 220,   0),  # TR — cyan
    (0,    50, 220),  # BR — red
    (220,  50,   0),  # BL — blue
]
POINT_RADIUS = 10
DRAG_RADIUS  = 18   # how close cursor must be to start a drag


class CornerEditor:
    def __init__(self, frame, corners):
        self.cv2     = _require("cv2", "opencv-python")
        self.np      = _require("numpy")
        self.orig    = frame.copy()
        self.corners = [list(c) for c in corners]  # [[x,y], ...]
        self.h, self.w = frame.shape[:2]

        # Viewport state (for zoom/pan)
        self.zoom    = 1.0
        self.pan_x   = 0.0   # offset in original image pixels
        self.pan_y   = 0.0
        self.zoom_min = 0.5
        self.zoom_max = 8.0

        # Interaction state
        self.dragging     = None   # index of corner being dragged, or None
        self.panning      = False
        self.pan_start    = None   # (mouse_x, mouse_y) when pan started
        self.pan_origin   = None   # (pan_x, pan_y) when pan started
        self.saved        = False
        self.quit_nosave  = False

        self.win = "FTC Field Corner Calibrator"
        self.cv2.namedWindow(self.win, self.cv2.WINDOW_NORMAL)
        self.cv2.resizeWindow(self.win, min(1280, self.w * 2), min(800, self.h * 2))
        self.cv2.setMouseCallback(self.win, self._mouse_cb)

    # ── coordinate helpers ────────────────────────────────────────
    def _img_to_view(self, ix, iy):
        """Image pixel → display pixel."""
        vx = (ix - self.pan_x) * self.zoom
        vy = (iy - self.pan_y) * self.zoom
        return int(vx), int(vy)

    def _view_to_img(self, vx, vy):
        """Display pixel → image pixel."""
        ix = vx / self.zoom + self.pan_x
        iy = vy / self.zoom + self.pan_y
        return ix, iy

    def _clamp_corner(self, x, y):
        return max(0, min(self.w - 1, x)), max(0, min(self.h - 1, y))

    # ── mouse callback ────────────────────────────────────────────
    def _mouse_cb(self, event, vx, vy, flags, param):
        cv2 = self.cv2
        ix, iy = self._view_to_img(vx, vy)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking near a corner
            best, best_d = None, float("inf")
            for i, (cx, cy) in enumerate(self.corners):
                d = math.hypot(ix - cx, iy - cy)
                if d < best_d:
                    best_d, best = d, i
            if best is not None and best_d < DRAG_RADIUS / self.zoom:
                self.dragging = best

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging is not None:
                x, y = self._clamp_corner(int(ix), int(iy))
                self.corners[self.dragging] = [x, y]
            elif self.panning and self.pan_start is not None:
                dx = (vx - self.pan_start[0]) / self.zoom
                dy = (vy - self.pan_start[1]) / self.zoom
                self.pan_x = self.pan_origin[0] - dx
                self.pan_y = self.pan_origin[1] - dy

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.panning   = True
            self.pan_start  = (vx, vy)
            self.pan_origin = (self.pan_x, self.pan_y)

        elif event == cv2.EVENT_RBUTTONUP:
            self.panning = False

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.panning   = True
            self.pan_start  = (vx, vy)
            self.pan_origin = (self.pan_x, self.pan_y)

        elif event == cv2.EVENT_MBUTTONUP:
            self.panning = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom centred on cursor
            factor = 1.15 if flags > 0 else 1 / 1.15
            new_zoom = max(self.zoom_min, min(self.zoom_max, self.zoom * factor))
            # Keep the image point under the cursor fixed
            self.pan_x = ix - vx / new_zoom
            self.pan_y = iy - vy / new_zoom
            self.zoom  = new_zoom

    # ── render ────────────────────────────────────────────────────
    def _render(self):
        cv2, np = self.cv2, self.np
        # Build zoomed/panned view
        view_h = int(self.h * self.zoom)
        view_w = int(self.w * self.zoom)
        # Determine the crop from orig that maps to the view window
        # We'll just resize a subregion for efficiency
        display = cv2.resize(self.orig, (view_w, view_h), interpolation=cv2.INTER_LINEAR)

        # Draw filled quadrilateral (semi-transparent)
        pts_view = np.array([self._img_to_view(*c) for c in self.corners], dtype=np.int32)
        overlay  = display.copy()
        cv2.fillPoly(overlay, [pts_view], (80, 80, 80))
        cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)

        # Draw edges
        for i in range(4):
            p1 = self._img_to_view(*self.corners[i])
            p2 = self._img_to_view(*self.corners[(i + 1) % 4])
            cv2.line(display, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw corner points and labels
        for i, (cx, cy) in enumerate(self.corners):
            vp = self._img_to_view(cx, cy)
            color = COLORS[i]
            # Outer white ring
            cv2.circle(display, vp, POINT_RADIUS + 3, (255, 255, 255), 2, cv2.LINE_AA)
            # Colored fill
            cv2.circle(display, vp, POINT_RADIUS, color, -1, cv2.LINE_AA)
            # Label above the point
            lx = vp[0] - 14
            ly = vp[1] - POINT_RADIUS - 8
            cv2.putText(display, LABELS[i], (lx + 1, ly + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            cv2.putText(display, LABELS[i], (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            # Pixel coordinates
            coord_str = f"({cx},{cy})"
            cv2.putText(display, coord_str, (vp[0] + POINT_RADIUS + 4, vp[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # HUD
        lines = [
            "Drag corners to field boundaries",
            "Scroll to zoom  |  Right-drag to pan",
            "R = reset  |  ENTER/S = save  |  ESC/Q = quit",
            f"Zoom: {self.zoom:.1f}x",
        ]
        for j, line in enumerate(lines):
            y = 22 + j * 22
            cv2.putText(display, line, (11, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            cv2.putText(display, line, (10, y),     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 180), 1)

        return display

    # ── main loop ─────────────────────────────────────────────────
    def run(self):
        cv2 = self.cv2
        print("[INFO] Window open. Drag the 4 colored corners onto the field boundary corners.")
        print("       TL=yellow  TR=cyan  BR=red  BL=blue")
        print("       Press ENTER or S to save, ESC or Q to quit without saving.")

        while True:
            frame = self._render()
            cv2.imshow(self.win, frame)
            key = cv2.waitKey(16) & 0xFF

            if key in (13, ord('s'), ord('S')):   # ENTER or S
                self.saved = True
                break
            elif key in (27, ord('q'), ord('Q')): # ESC or Q
                self.quit_nosave = True
                break
            elif key in (ord('r'), ord('R')):
                self.corners = default_corners(self.w, self.h)
                print("[INFO] Corners reset to defaults.")
            elif key == ord('+') or key == ord('='):
                self.zoom = min(self.zoom_max, self.zoom * 1.2)
            elif key == ord('-'):
                self.zoom = max(self.zoom_min, self.zoom / 1.2)

            # Check if window was closed
            if cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:
                self.quit_nosave = True
                break

        cv2.destroyAllWindows()
        return self.corners if self.saved else None


def main():
    parser = argparse.ArgumentParser(
        description="Interactive FTC field corner picker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", help="Path to the match video file")
    parser.add_argument("--output", "-o", default="field_corners.json",
                        help="Output JSON file path (default: field_corners.json)")
    parser.add_argument("--frame", "-f", type=int, default=None,
                        help="Frame number to display (default: middle frame)")
    args = parser.parse_args()

    frame = get_middle_frame(args.video, args.frame)
    h, w  = frame.shape[:2]
    corners = default_corners(w, h)

    editor  = CornerEditor(frame, corners)
    result  = editor.run()

    if result is None:
        print("[INFO] Cancelled — no file saved.")
        sys.exit(0)

    # result is [TL, TR, BR, BL] in image pixel coords
    tl, tr, br, bl = result

    # Save in the format auto_scout.py --corners expects:
    # {"corners_px": [bl, br, tr, tl]}
    data = {"corners_px": [bl, br, tr, tl]}
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n[SAVED] {args.output}")
    print(f"  TL = {tl}")
    print(f"  TR = {tr}")
    print(f"  BR = {br}")
    print(f"  BL = {bl}")
    print(f"\nRun your tracker with:")
    print(f"  python3 auto_scout.py --no-download --video-path <video> --corners {args.output}")


if __name__ == "__main__":
    main()