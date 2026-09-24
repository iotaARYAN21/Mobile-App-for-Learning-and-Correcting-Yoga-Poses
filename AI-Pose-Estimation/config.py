# config.py — All thresholds with justification
# Exercise: Urdhva Hastasana (Raised Hands Pose)

# ── Landmark indices (MediaPipe BlazePose 33-point model) ──────────────────
LANDMARKS = {
    "NOSE": 0,
    "LEFT_EAR": 7, "RIGHT_EAR": 8,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
}

# ── Pose correction thresholds ─────────────────────────────────────────────

# Check 1: Feet ~15 cm apart
# Source: Exercise text says "6 inches (15 cm) apart"
# Tolerance ±5 cm for natural variation and sensor noise
FEET_DIST_MIN_CM = 10
FEET_DIST_MAX_CM = 20

# Check 2: Arms fully stretched (elbow angle)
# Source: Physiotherapy defines functional full extension as >160°
# 180° is theoretical max; 160° accounts for natural hyperextension variation
ELBOW_ANGLE_MIN = 160.0

# Check 3: Arms close to ears
# Source: Exercise text says "arms should be close to the ears"
# Threshold = 12% of frame width; wrists should be nearly above ears
ARM_EAR_THRESHOLD = 0.12  # fraction of frame width

# Check 4: Palms joined
# Source: Exercise text says "join the palms and fingers together"
# Threshold = 6% of frame width; pressed palms still have ~5cm natural spread
PALMS_JOINED_THRESHOLD = 0.06  # fraction of frame width

# Check 5: Hold for 4 breaths
# Source: Exercise text says "four complete breaths"
# Average adult breath cycle = 4 seconds → 4 × 4 = 16 seconds
BREATH_DURATION = 4.0   # seconds per breath
HOLD_BREATHS = 4
HOLD_DURATION = BREATH_DURATION * HOLD_BREATHS  # 16 seconds

# Check 6: Rest phase — 2 breaths
# Source: Exercise text says "rest for two complete breaths"
REST_BREATHS = 2
REST_DURATION = BREATH_DURATION * REST_BREATHS   # 8 seconds
REST_TOLERANCE = 0.05  # normalized units; wrists at/below hip level

# ── Visibility filter ──────────────────────────────────────────────────────
# Skip landmark if confidence below this — prevents spurious corrections
MIN_VISIBILITY = 0.5

# ── Display ────────────────────────────────────────────────────────────────
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_BLUE = (255, 180, 0)

# ── Full-body visibility detection (visibility.py) ─────────────────────────
# A landmark counts as "seen" only if MediaPipe's confidence is at least this.
# MediaPipe still outputs *guessed* coordinates for joints outside the frame,
# but with low visibility, so confidence + position must BOTH pass.
BODY_LANDMARK_MIN_VISIBILITY = 0.6

# Every required landmark must lie inside the frame, at least this fraction of
# the frame size away from each edge (2% ≈ 26 px on a 1280-wide frame).
BODY_EDGE_MARGIN = 0.02

# Head/feet span must cover at least this fraction of frame height, otherwise
# the person is too far away for reliable landmarks.
BODY_MIN_HEIGHT_FRAC = 0.30

# Debounce (in frames): quick to say "not visible", slower to say "visible",
# so the label doesn't flicker when a joint hovers near an edge.
BODY_HIDDEN_AFTER_FRAMES = 3
BODY_VISIBLE_AFTER_FRAMES = 8
