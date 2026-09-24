# main.py — Yoga Pose Correction Engine
# Exercise: Urdhva Hastasana (Raised Hands Pose)
# Run with: python3.11 main.py

import cv2
import time

from pose_engine import create_pose_detector, process_frame, get_landmarks, get_world_landmarks, open_webcam
from correction import evaluate_all
from utils import (FPSCounter, draw_status_panel, draw_angle_label,
                    draw_feedback_banner, draw_coordinates, calculate_angle,
                    draw_body_visibility)
from visibility import BodyVisibilityDetector
from config import *

WINDOW_NAME = "Yoga Pose Correction — Urdhva Hastasana"


def get_primary_feedback(checks, messages, phase):
    """Return the most important feedback message to show in the banner."""
    if phase == "COMPLETE":
        return "Set Complete! Lower arms and rest.", COLOR_GREEN

    if phase == "REST":
        return messages[5], COLOR_ORANGE

    if phase == "HOLD":
        if all(checks[:4]):
            return messages[4], COLOR_GREEN
        # Show first failing spatial check
        for i in range(4):
            if not checks[i]:
                return messages[i], COLOR_RED

    # SETUP phase — guide user into position
    for i in range(4):
        if not checks[i]:
            return messages[i], COLOR_RED

    return "Great posture! Hold it.", COLOR_GREEN


def main():
    print("=" * 55)
    print(" Yoga Pose Correction Engine")
    print(" Exercise: Urdhva Hastasana (Raised Hands Pose)")
    print("=" * 55)
    print("Controls: Q = quit | R = reset rep counter")
    print()

    cap = open_webcam(cam_index=0, width=1280, height=720)
    fps_counter = FPSCounter()

    # Fix: create a resizable window up front. cv2.imshow() with no
    # prior cv2.namedWindow() call defaults to WINDOW_AUTOSIZE, which
    # shrinks the window to match whatever resolution the camera
    # actually delivers (often much smaller than requested). Using
    # WINDOW_NORMAL + resizeWindow decouples the display size from
    # the camera's native frame size and lets the window be resized
    # by hand too.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    # State
    hold_timer = 0.0
    rest_timer = 0.0
    rep_count = 0
    phase = "SETUP"  # SETUP → HOLD → REST → COMPLETE → SETUP
    in_hold_phase = False
    set_complete = False
    prev_time = time.time()

    pose = create_pose_detector()
    body_detector = BodyVisibilityDetector()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Flip for mirror view (more intuitive for user)
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = fps_counter.update()

            # ── Detection ─────────────────────────────────────────────────
            results = process_frame(pose, frame)
            landmarks = get_landmarks(results)
            world_landmarks = get_world_landmarks(results)

            # ── Full-body visibility (head → toes inside the frame?) ──────
            body_vis = body_detector.update(landmarks)
            if body_vis.changed:
                detail = "" if body_vis.visible else " — " + "; ".join(body_vis.reasons)
                print(f"[visibility] {body_vis.label}{detail}")

            # Default checks (all False) when no person detected
            checks = [False] * 6
            messages = [""] * 6
            spatial_ok = False
            arms_down = False

            if landmarks and body_vis.visible:
                checks, messages, spatial_ok, arms_down = evaluate_all(
                    landmarks, world_landmarks, w,
                    hold_timer, rest_timer, in_hold_phase
                )

                # ── State Machine ──────────────────────────────────────────
                if phase == "SETUP":
                    if spatial_ok:
                        phase = "HOLD"
                        hold_timer = 0.0

                elif phase == "HOLD":
                    if spatial_ok:
                        hold_timer += dt
                        if hold_timer >= HOLD_DURATION:
                            phase = "REST"
                            rest_timer = 0.0
                    else:
                        hold_timer = max(0.0, hold_timer - dt * 0.5)  # slow decay

                elif phase == "REST":
                    if arms_down:
                        rest_timer += dt
                        if rest_timer >= REST_DURATION:
                            rep_count += 1
                            if rep_count >= 3:
                                phase = "COMPLETE"
                            else:
                                phase = "SETUP"
                            hold_timer = 0.0
                            rest_timer = 0.0
                    else:
                        rest_timer = max(0.0, rest_timer - dt * 0.3)

                # Draw angle labels on elbows
                lms = landmarks
                l_angle = calculate_angle(
                    lms[LANDMARKS["LEFT_SHOULDER"]],
                    lms[LANDMARKS["LEFT_ELBOW"]],
                    lms[LANDMARKS["LEFT_WRIST"]]
                )
                r_angle = calculate_angle(
                    lms[LANDMARKS["RIGHT_SHOULDER"]],
                    lms[LANDMARKS["RIGHT_ELBOW"]],
                    lms[LANDMARKS["RIGHT_WRIST"]]
                )
                draw_angle_label(frame, lms[LANDMARKS["LEFT_ELBOW"]], l_angle, w, h)
                draw_angle_label(frame, lms[LANDMARKS["RIGHT_ELBOW"]], r_angle, w, h)

                # Coordinates panel
                draw_coordinates(frame, lms, w, h)
            elif phase == "HOLD":
                # Body left the frame mid-hold: progress decays, same as a bad pose
                hold_timer = max(0.0, hold_timer - dt * 0.5)

            # ── UI ─────────────────────────────────────────────────────────
            # Status panel (right side)
            draw_status_panel(frame, checks, hold_timer, rest_timer, phase, fps)

            # Primary feedback banner (top left)
            if body_vis.visible:
                banner_msg, banner_color = get_primary_feedback(checks, messages, phase)
            else:
                banner_msg = body_vis.hint or "Get your whole body in view"
                banner_color = COLOR_YELLOW
            draw_feedback_banner(frame, banner_msg, banner_color)
            draw_body_visibility(frame, body_vis)

            # Rep counter
            cv2.putText(frame, f"Reps: {rep_count}/3", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_YELLOW, 2)

            # Phase label
            cv2.putText(frame, f"[{phase}]", (20, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)

            # Print coordinates to terminal
            if landmarks and int(fps_counter.fps) % 30 == 0:
                lms = landmarks
                print(f"[FPS:{fps:.1f}] Phase:{phase} | "
                      f"LWrist:({lms[15].x:.2f},{lms[15].y:.2f}) "
                      f"RWrist:({lms[16].x:.2f},{lms[16].y:.2f}) "
                      f"Hold:{hold_timer:.1f}s Rest:{rest_timer:.1f}s Reps:{rep_count}")

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset
                hold_timer = 0.0
                rest_timer = 0.0
                rep_count = 0
                phase = "SETUP"
                body_detector.reset()
                print("Reset!")

    finally:
        pose.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession ended. Reps completed: {rep_count}/3")


if __name__ == "__main__":
    main()
