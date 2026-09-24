# correction.py — Rule-based pose correction engine
# All 6 checks for Urdhva Hastasana (Raised Hands Pose)

from config import *
from utils import (calculate_angle, landmark_visible,
                   pixel_distance, world_distance_cm)


def check_feet_apart(landmarks, world_landmarks):
    """
    Check 1: Feet ~15 cm apart.

    Strategy: Use ankle/hip pixel ratio as a camera-distance-independent measure.
    The average adult hip width is ~35 cm. The exercise requires 15 cm foot gap.
    So the ankle separation should be roughly 0.3–0.7x the hip width.

    We also try world landmarks as a secondary check, but pixel ratio is primary
    because world landmark depth estimation is unreliable at typical webcam distances.
    """
    lms = landmarks

    la = lms[LANDMARKS["LEFT_ANKLE"]]
    ra = lms[LANDMARKS["RIGHT_ANKLE"]]
    lh = lms[LANDMARKS["LEFT_HIP"]]
    rh = lms[LANDMARKS["RIGHT_HIP"]]

    if not all(landmark_visible(x) for x in [la, ra, lh, rh]):
        return False, "Ankles/hips not visible"

    # Pixel-based ratio: ankle_gap / hip_width
    ankle_gap = abs(la.x - ra.x)
    hip_width  = abs(lh.x - rh.x)

    if hip_width < 0.01:
        return False, "Cannot measure — stand facing camera"

    ratio = ankle_gap / hip_width

    # DEBUG — print live ratio so we can tune the threshold
    print(f"[FEET DEBUG] ankle_gap={ankle_gap:.3f}  hip_width={hip_width:.3f}  ratio={ratio:.3f}")

    # Very loose thresholds — adjust after seeing debug output
    RATIO_MIN = 1.0
    RATIO_MAX = 2.0

    ok = RATIO_MIN <= ratio <= RATIO_MAX

    # Also try world landmarks for display info
    world_str = ""
    if world_landmarks:
        try:
            wla = world_landmarks[LANDMARKS["LEFT_ANKLE"]]
            wra = world_landmarks[LANDMARKS["RIGHT_ANKLE"]]
            world_dist = world_distance_cm(wla, wra)
            world_str = f" (~{world_dist:.0f}cm)"
        except Exception:
            pass

    if ratio < RATIO_MIN:
        msg = f"Feet too close{world_str} — spread apart slightly"
    elif ratio > RATIO_MAX:
        msg = f"Feet too wide{world_str} — bring feet closer"
    else:
        msg = f"Feet OK{world_str} (ratio:{ratio:.2f})"

    return ok, msg


def check_arms_stretched(landmarks):
    """
    Check 2: Elbow angle > 160° on both sides (arms fully extended).
    Threshold: 160° — physiotherapy definition of functional full extension.
    """
    lms = landmarks

    ls = lms[LANDMARKS["LEFT_SHOULDER"]]
    le = lms[LANDMARKS["LEFT_ELBOW"]]
    lw = lms[LANDMARKS["LEFT_WRIST"]]
    rs = lms[LANDMARKS["RIGHT_SHOULDER"]]
    re = lms[LANDMARKS["RIGHT_ELBOW"]]
    rw = lms[LANDMARKS["RIGHT_WRIST"]]

    # Visibility filter
    if not all(landmark_visible(x) for x in [ls, le, lw, rs, re, rw]):
        return False, "Move into frame — arms not fully visible"

    l_angle = calculate_angle(ls, le, lw)
    r_angle = calculate_angle(rs, re, rw)

    if l_angle is None or r_angle is None:
        return False, "Cannot compute arm angle"

    l_ok = l_angle > ELBOW_ANGLE_MIN
    r_ok = r_angle > ELBOW_ANGLE_MIN

    if l_ok and r_ok:
        return True, f"Arms straight (L:{int(l_angle)}° R:{int(r_angle)}°)"
    elif not l_ok and not r_ok:
        return False, f"Straighten both arms (L:{int(l_angle)}° R:{int(r_angle)}°)"
    elif not l_ok:
        return False, f"Straighten LEFT arm ({int(l_angle)}°)"
    else:
        return False, f"Straighten RIGHT arm ({int(r_angle)}°)"


def check_arms_near_ears(landmarks, frame_w):
    """
    Check 3: Wrists horizontally aligned with ears.
    Threshold: horizontal distance < 12% of frame width.
    """
    lms = landmarks
    le = lms[LANDMARKS["LEFT_EAR"]]
    re = lms[LANDMARKS["RIGHT_EAR"]]
    lw = lms[LANDMARKS["LEFT_WRIST"]]
    rw = lms[LANDMARKS["RIGHT_WRIST"]]

    if not all(landmark_visible(x) for x in [le, re, lw, rw]):
        return False, "Ears or wrists not visible"

    l_dist = abs(lw.x - le.x) * frame_w
    r_dist = abs(rw.x - re.x) * frame_w
    threshold = ARM_EAR_THRESHOLD * frame_w

    l_ok = l_dist < threshold
    r_ok = r_dist < threshold

    if l_ok and r_ok:
        return True, "Arms close to ears ✓"
    elif not l_ok and not r_ok:
        return False, "Bring BOTH arms closer to ears"
    elif not l_ok:
        return False, "Bring LEFT arm closer to ear"
    else:
        return False, "Bring RIGHT arm closer to ear"


def check_palms_joined(landmarks, frame_w):
    """
    Check 4: Both wrists close together (palms joined).
    Threshold: wrist distance < 6% of frame width.
    """
    lms = landmarks
    lw = lms[LANDMARKS["LEFT_WRIST"]]
    rw = lms[LANDMARKS["RIGHT_WRIST"]]

    if not (landmark_visible(lw) and landmark_visible(rw)):
        return False, "Wrists not visible"

    dist = abs(lw.x - rw.x) * frame_w
    threshold = PALMS_JOINED_THRESHOLD * frame_w

    ok = dist < threshold
    msg = "Palms joined ✓" if ok else f"Join your palms (gap: {int(dist)}px)"
    return ok, msg


def check_hold_timer(hold_timer):
    """
    Check 5: Posture held for 4 complete breaths (16 seconds).
    Timer managed externally; this just evaluates completion.
    """
    ok = hold_timer >= HOLD_DURATION
    breaths = min(int(hold_timer / BREATH_DURATION), HOLD_BREATHS)
    msg = (f"Hold complete! ({HOLD_BREATHS} breaths)" if ok
           else f"Hold... {breaths}/{HOLD_BREATHS} breaths ({hold_timer:.1f}s)")
    return ok, msg


def check_rest_phase(landmarks, rest_timer):
    """
    Check 6: Arms lowered to thighs (wrists at/below hip level) for 2 breaths.
    y increases downward in image coords — wrist.y >= hip.y means arms are down.
    """
    lms = landmarks
    lw = lms[LANDMARKS["LEFT_WRIST"]]
    rw = lms[LANDMARKS["RIGHT_WRIST"]]
    lh = lms[LANDMARKS["LEFT_HIP"]]
    rh = lms[LANDMARKS["RIGHT_HIP"]]

    if not all(landmark_visible(x) for x in [lw, rw, lh, rh]):
        return False, False, "Hips/wrists not visible"

    arms_down = (lw.y >= lh.y - REST_TOLERANCE and
                 rw.y >= rh.y - REST_TOLERANCE)

    rest_complete = rest_timer >= REST_DURATION
    breaths = min(int(rest_timer / BREATH_DURATION), REST_BREATHS)

    if rest_complete:
        msg = "Rest complete! Ready for next rep."
    elif arms_down:
        msg = f"Resting... {breaths}/{REST_BREATHS} breaths ({rest_timer:.1f}s)"
    else:
        msg = "Lower arms to thighs to rest"

    return arms_down, rest_complete, msg


def evaluate_all(landmarks, world_landmarks, frame_w,
                 hold_timer, rest_timer, in_hold_phase):
    """
    Run all 6 checks and return:
      - checks: list of 6 booleans
      - messages: list of 6 feedback strings
      - overall_ok: True if checks 1-4 all pass (spatial checks)
    """
    c1, m1 = check_feet_apart(landmarks, world_landmarks)
    c2, m2 = check_arms_stretched(landmarks)
    c3, m3 = check_arms_near_ears(landmarks, frame_w)
    c4, m4 = check_palms_joined(landmarks, frame_w)
    c5, m5 = check_hold_timer(hold_timer)

    arms_down, rest_complete, m6 = check_rest_phase(landmarks, rest_timer)
    c6 = rest_complete

    checks = [c1, c2, c3, c4, c5, c6]
    messages = [m1, m2, m3, m4, m5, m6]
    spatial_ok = c1 and c2 and c3 and c4   # all position checks pass

    return checks, messages, spatial_ok, arms_down
