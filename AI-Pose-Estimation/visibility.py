# visibility.py — Is the WHOLE body inside the camera frame?
#
# Two conditions must hold for every required landmark (head → toes):
#   1. MediaPipe confidence (landmark.visibility) >= BODY_LANDMARK_MIN_VISIBILITY
#   2. Its (x, y) lies inside the frame, BODY_EDGE_MARGIN away from every edge
# MediaPipe has no "top of head" landmark, so it is estimated from the nose and
# shoulder line. The result is debounced so the label doesn't flicker.

from dataclasses import dataclass, field
from typing import List

from config import (BODY_LANDMARK_MIN_VISIBILITY, BODY_EDGE_MARGIN,
                    BODY_MIN_HEIGHT_FRAC, BODY_HIDDEN_AFTER_FRAMES,
                    BODY_VISIBLE_AFTER_FRAMES)

# group name -> BlazePose landmark indices that must be in view
REQUIRED_GROUPS = {
    "Head":      [0],            # nose (+ estimated top of head, see below)
    "Shoulders": [11, 12],
    "Elbows":    [13, 14],
    "Hands":     [15, 16],       # wrists
    "Hips":      [23, 24],
    "Knees":     [25, 26],
    "Feet":      [27, 28, 29, 30, 31, 32],   # ankles, heels, toe tips
}
_FOOT_IDS = (27, 28, 29, 30, 31, 32)
_HEAD_TOP_FACTOR = 0.6   # head-top ≈ nose - 0.6 × (nose → shoulder-line distance)


@dataclass
class BodyVisibility:
    visible: bool                      # debounced, final answer
    raw_visible: bool                  # this frame only, no debouncing
    reasons: List[str] = field(default_factory=list)
    hint: str = ""
    changed: bool = False              # True on the frame `visible` flipped

    @property
    def label(self):
        return "BODY VISIBLE" if self.visible else "BODY NOT VISIBLE"


def _edge_of(x, y, m):
    """Which frame edge a point is beyond ('' if inside)."""
    if y < m:
        return "top"
    if y > 1 - m:
        return "bottom"
    if x < m:
        return "left"
    if x > 1 - m:
        return "right"
    return ""


def analyse_frame(landmarks, margin=BODY_EDGE_MARGIN,
                  min_vis=BODY_LANDMARK_MIN_VISIBILITY,
                  min_height=BODY_MIN_HEIGHT_FRAC):
    """
    Single-frame check (no debouncing). Returns (ok, reasons, hint).
    `landmarks` is MediaPipe's normalised landmark list, or None.
    """
    if not landmarks:
        return False, ["No person detected"], "Stand in front of camera"

    reasons, edges, low_conf = [], set(), False

    for group, ids in REQUIRED_GROUPS.items():
        out_edges = {e for i in ids
                     if (e := _edge_of(landmarks[i].x, landmarks[i].y, margin))}
        unsure = any(landmarks[i].visibility < min_vis for i in ids)
        if out_edges:
            edges |= out_edges
            reasons.append(f"{group} out of frame ({'/'.join(sorted(out_edges))})")
        elif unsure:
            low_conf = True
            reasons.append(f"{group} not clearly visible")

    # Top of head has no landmark: estimate it from nose + shoulder line.
    nose = landmarks[0]
    sh_y = (landmarks[11].y + landmarks[12].y) / 2
    head_top_y = nose.y - _HEAD_TOP_FACTOR * max(sh_y - nose.y, 0.05)
    if head_top_y < margin and not any(r.startswith("Head out") for r in reasons):
        edges.add("top")
        reasons.append("Head out of frame (top)")

    # Only judge distance once everything else is fine.
    if not reasons:
        bottom_y = max(landmarks[i].y for i in _FOOT_IDS)
        if bottom_y - head_top_y < min_height:
            reasons.append("Too far from camera")
            return False, reasons, "Move closer to the camera"
        return True, [], ""

    if edges & {"top", "bottom"}:
        hint = "Step back so your whole body fits"
    elif edges & {"left", "right"}:
        hint = "Move toward the center of the frame"
    elif low_conf:
        hint = "Face the camera in good lighting"
    else:
        hint = "Adjust position"
    return False, reasons, hint


class BodyVisibilityDetector:
    """Debounced wrapper around analyse_frame()."""

    def __init__(self):
        self.visible = False
        self._ok_streak = 0
        self._bad_streak = 0

    def reset(self):
        self.__init__()

    def update(self, landmarks) -> BodyVisibility:
        ok, reasons, hint = analyse_frame(landmarks)
        if ok:
            self._ok_streak += 1
            self._bad_streak = 0
        else:
            self._bad_streak += 1
            self._ok_streak = 0

        prev = self.visible
        if not self.visible and self._ok_streak >= BODY_VISIBLE_AFTER_FRAMES:
            self.visible = True
        elif self.visible and self._bad_streak >= BODY_HIDDEN_AFTER_FRAMES:
            self.visible = False

        if self.visible:
            reasons, hint = [], ""           # a 1-2 frame glitch is ignored
        elif ok:
            reasons, hint = ["Confirming..."], "Hold still"   # good, not yet stable

        return BodyVisibility(self.visible, ok, reasons, hint,
                              changed=(self.visible != prev))
