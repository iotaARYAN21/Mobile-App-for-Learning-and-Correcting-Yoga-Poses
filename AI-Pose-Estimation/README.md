# Yoga Pose Estimation & Correction Engine

A real-time yoga pose correction system built with **MediaPipe** and **OpenCV** for the internship technical assignment. Detects and corrects **Urdhva Hastasana (Raised Hands Pose)** using rule-based logic on 33 body landmarks — no model training required.

---

## Demo

```
Stand erect → feet 6 inches apart → raise arms above head → join palms
→ hold for 4 breaths → lower arms to thighs → rest for 2 breaths → repeat 3x
```

The system gives real-time feedback on each condition, tracks breath hold timers, and counts reps automatically.

---

## Features

- **Live webcam pose detection** at 25–30 FPS on CPU (no GPU required)
- **6 rule-based pose checks** running simultaneously
- **Camera-distance independent** — all checks normalized by shoulder width
- **Body-size independent** — works for any person regardless of height
- **Breath timer** with visual progress bar
- **Rep counter** tracking 3 complete reps
- **Joint angle display** overlaid on skeleton
- **Landmark coordinates** printed to screen and terminal
- **Modular codebase** — clean separation of detection, correction, and UI

---

## Pose Checks

| # | Check | Method | Threshold |
|---|-------|--------|-----------|
| 1 | Feet ~15 cm apart | `ankle_gap / shoulder_width` ratio | 0.22 – 0.55 |
| 2 | Arms fully stretched | Elbow angle (both sides) | > 160° |
| 3 | Arms close to ears | `wrist-ear offset / shoulder_width` | < 0.30 |
| 4 | Palms joined | `wrist_gap / shoulder_width` | < 0.20 |
| 5 | Hold for 4 breaths | Timer (4 breaths × 4 sec = 16 sec) | ≥ 16 seconds |
| 6 | Rest for 2 breaths | Wrists at/below hip level for 8 sec | ≥ 8 seconds |

All spatial checks use **shoulder width as the normalization reference**, making them invariant to camera distance and body size.

---

## Full-Body Visibility Detection

The screen always shows **BODY VISIBLE** (green) or **BODY NOT VISIBLE** (red) with the reasons
(e.g. `Feet out of frame (bottom)`, `Hands out of frame (top)`, `Too far from camera`).
The body counts as visible only if every landmark from head to toes is (1) above the
MediaPipe confidence threshold **and** (2) inside the frame with a small edge margin
(MediaPipe guesses positions for joints that are off-screen, so confidence alone isn't enough).
The top of the head is estimated from the nose and shoulders. Pose checks and timers only run
while the body is fully visible. Tune `BODY_*` values in `config.py`.

---

## Project Structure

```
├── main.py           # Entry point — state machine & orchestration
├── pose_engine.py    # MediaPipe pose detection & webcam management
├── correction.py     # Rule-based pose correction logic (6 checks)
├── visibility.py     # Full-body-in-frame detection (BODY VISIBLE / NOT VISIBLE)
├── utils.py          # Angle calculation, drawing helpers, FPS counter
├── config.py         # All thresholds with justification comments
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python **3.11** (required — MediaPipe `solutions` API is not available on Python 3.12+)
- Webcam

### Install Python 3.11 (Mac)
```bash
brew install python@3.11
```

### Clone & Install Dependencies
```bash
git clone https://github.com/YOUR_USERNAME/yoga-pose-correction.git
cd yoga-pose-correction

python3.11 -m pip install -r requirements.txt
```

### Run
```bash
python3.11 main.py
```

> **Mac users:** Grant camera permission to Terminal via  
> System Settings → Privacy & Security → Camera → Terminal ✓

---

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset rep counter and timers |

---

## How It Works

### Framework: MediaPipe BlazePose
MediaPipe was selected over MoveNet, OpenPose, and Detectron2 for:
- **33 landmarks** including ear landmarks (required for arm-to-ear check)
- **25–30 FPS on CPU** — no GPU needed
- **3D world coordinates** in metric units
- Simple pip install, Apache 2.0 license

### Pose Detection Pipeline
```
Webcam frame → BGR→RGB → MediaPipe Pose → 33 landmarks (x, y, z, visibility)
    → Correction Engine → 6 rule checks → Feedback overlay → Display
```

### Angle Calculation
Joint angles use the **dot product formula**:
```
BA = A - B,  BC = C - B
angle = arccos( (BA · BC) / (|BA| × |BC|) )
```
Applied to (SHOULDER, ELBOW, WRIST) triplets to detect arm extension.

### Scale Invariance
All distance-based checks divide by shoulder width:
```
ratio = ankle_gap / shoulder_width
```
Since both values scale identically with camera distance and body size, the ratio is a pure measure of posture — valid for any person at any distance.

### State Machine
```
SETUP → (all 4 spatial checks pass) → HOLD → (16s timer) → REST → (8s timer) → SETUP
                                                                  ↓ (after 3 reps)
                                                               COMPLETE
```

---

## Threshold Justification

| Threshold | Value | Justification |
|-----------|-------|---------------|
| Feet ratio min | 0.22 | 15cm/40cm = 0.375 expected; −40% tolerance |
| Feet ratio max | 0.55 | 15cm/40cm = 0.375 expected; +40% tolerance |
| Elbow angle | 160° | Physiotherapy definition of functional full extension |
| Arm-ear offset | 0.30× shoulder | Wrist directly above ear = 0; 0.30 allows natural variation |
| Palm gap | 0.20× shoulder | Pressed palms still have ~5cm natural spread |
| Hold duration | 16 sec | 4 breaths × 4 sec/breath (average adult resting rate) |
| Rest duration | 8 sec | 2 breaths × 4 sec/breath |
| Min visibility | 0.50 | MediaPipe confidence filter before computing angles |

---

## Review Questions (Assignment)

**Q1. How does MediaPipe detect joints internally?**  
Two-stage pipeline: (1) BlazeFace-based person detector estimates bounding ROI on first frame; (2) BlazePose landmark model regresses 33 3D coordinates directly inside the ROI. Subsequent frames reuse the previous ROI via a tracker — the detector only re-runs if tracking confidence drops below threshold.

**Q2. Why is MediaPipe suitable for this yoga use case?**  
It detects 33 landmarks (including ears — required for arm-to-ear check), runs at 25–30 FPS on CPU, provides 3D world coordinates for metric measurements, and has published 100% posture classification accuracy on yoga poses.

**Q3. How are angles computed?**  
Dot product formula: `arccos((BA · BC) / (|BA| × |BC|))` with `np.clip` to prevent domain errors. Applied to (SHOULDER, ELBOW, WRIST) triplets for elbow angle detection.

**Q4. Why those thresholds?**  
All thresholds are derived from anatomy (shoulder/foot proportions), the exercise specification (15 cm, 4 breaths), and physiotherapy standards (160° full extension). See table above.

**Q5. How to extend to multiple poses?**  
Add a `POSE_RULES` dictionary mapping pose names to rule sets. A lightweight classifier (kNN on joint angle feature vectors) identifies the current pose, then loads the corresponding rule set. No retraining of the CNN is needed.

---

## Dependencies

See `requirements.txt`. Core dependencies:
- `mediapipe==0.10.x` (install via pip under Python 3.11)
- `opencv-python`
- `numpy`

---

## License

MIT License — free to use, modify, and distribute.
