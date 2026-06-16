import math
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pygame


# ============================================================
# CONFIG
# ============================================================
WIDTH, HEIGHT = 1280, 780
PANEL_W = 380
CAM_W = WIDTH - PANEL_W
FPS = 60

BG = (11, 14, 24)
BG_2 = (16, 21, 36)
PANEL = (18, 24, 40)
CARD = (28, 35, 55)
CARD_2 = (34, 42, 66)
TEXT = (240, 244, 255)
MUTED = (170, 178, 198)
ACCENT = (76, 201, 240)
ACCENT_2 = (120, 180, 255)
GREEN = (46, 204, 113)
YELLOW = (241, 196, 15)
ORANGE = (243, 156, 18)
RED = (231, 76, 60)
WHITE = (255, 255, 255)
SHADOW = (0, 0, 0)


# ============================================================
# HELPERS
# ============================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(current, target, speed=0.18):
    return current + (target - current) * speed


def draw_round_rect(surface, color, rect, radius=18, border=0, border_color=None):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border > 0 and border_color is not None:
        pygame.draw.rect(surface, border_color, rect, width=border, border_radius=radius)


def draw_shadowed_text(surface, text, font, color, pos, shadow_offset=(2, 2), shadow_color=(0, 0, 0)):
    shadow = font.render(text, True, shadow_color)
    main = font.render(text, True, color)
    surface.blit(shadow, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
    surface.blit(main, pos)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ============================================================
# GESTURE LOGIC
# ============================================================
def count_fingers(hand_landmarks, handedness_label):
    """Count all 5 fingers using landmark geometry.

    Thumb uses horizontal comparison with handedness.
    Other fingers use tip.y < pip.y.
    """
    lm = hand_landmarks.landmark

    fingers = 0

    # Thumb
    # For a mirrored camera feed, MediaPipe handedness stays usable enough for this demo.
    # Right hand -> thumb is open when tip.x < ip.x
    # Left hand  -> thumb is open when tip.x > ip.x
    if handedness_label == "Right":
        if lm[4].x < lm[3].x:
            fingers += 1
    else:
        if lm[4].x > lm[3].x:
            fingers += 1

    # Index, middle, ring, pinky
    tip_ids = [8, 12, 16, 20]
    for tip in tip_ids:
        if lm[tip].y < lm[tip - 2].y:
            fingers += 1

    return fingers


def gesture_from_count(count):
    mapping = {
        0: ("FIST", "Hold Position / Stop"),
        1: ("ONE", "Move Left"),
        2: ("TWO", "Move Right"),
        3: ("THREE", "Move Up"),
        4: ("FOUR", "Move Down"),
        5: ("OPEN PALM", "Reset Arm"),
    }
    return mapping.get(count, ("UNKNOWN", "No Action"))


# ============================================================
# ROBOT ARM DRAWING
# ============================================================
def arm_forward_kinematics(base, l1, l2, l3, a1, a2, a3):
    x0, y0 = base
    r1 = math.radians(a1)
    r2 = math.radians(a1 + a2)
    r3 = math.radians(a1 + a2 + a3)

    x1 = x0 + l1 * math.cos(r1)
    y1 = y0 + l1 * math.sin(r1)

    x2 = x1 + l2 * math.cos(r2)
    y2 = y1 + l2 * math.sin(r2)

    x3 = x2 + l3 * math.cos(r3)
    y3 = y2 + l3 * math.sin(r3)

    return (x1, y1), (x2, y2), (x3, y3)


def draw_robot_arm(surface, base, joint1, joint2, gripper_open=26):
    # Realistic industrial-ish arm colors
    base_color = (52, 67, 88)
    arm_dark = (88, 120, 170)
    arm_light = (132, 183, 255)
    joint_color = (238, 240, 245)
    gripper_color = (210, 220, 235)
    highlight = (255, 255, 255)

    l1, l2, l3 = 190, 160, 85
    a1, a2, a3 = joint1, joint2, -joint2 * 0.55

    p1, p2, p3 = arm_forward_kinematics(base, l1, l2, l3, a1, a2, a3)

    x0, y0 = base
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Subtle shadow
    shadow_offset = 8
    pygame.draw.line(surface, (0, 0, 0), (x0 + shadow_offset, y0 + shadow_offset), (x1 + shadow_offset, y1 + shadow_offset), 22)
    pygame.draw.line(surface, (0, 0, 0), (x1 + shadow_offset, y1 + shadow_offset), (x2 + shadow_offset, y2 + shadow_offset), 18)
    pygame.draw.line(surface, (0, 0, 0), (x2 + shadow_offset, y2 + shadow_offset), (x3 + shadow_offset, y3 + shadow_offset), 16)

    # Arm segments
    pygame.draw.line(surface, arm_dark, (x0, y0), (x1, y1), 22)
    pygame.draw.line(surface, arm_light, (x0, y0), (x1, y1), 8)

    pygame.draw.line(surface, arm_dark, (x1, y1), (x2, y2), 18)
    pygame.draw.line(surface, arm_light, (x1, y1), (x2, y2), 7)

    pygame.draw.line(surface, arm_dark, (x2, y2), (x3, y3), 14)
    pygame.draw.line(surface, arm_light, (x2, y2), (x3, y3), 6)

    # Joints
    pygame.draw.circle(surface, base_color, (int(x0), int(y0)), 34)
    pygame.draw.circle(surface, joint_color, (int(x0), int(y0)), 16)

    pygame.draw.circle(surface, base_color, (int(x1), int(y1)), 22)
    pygame.draw.circle(surface, joint_color, (int(x1), int(y1)), 10)

    pygame.draw.circle(surface, base_color, (int(x2), int(y2)), 20)
    pygame.draw.circle(surface, joint_color, (int(x2), int(y2)), 9)

    # End effector + gripper
    grip_angle = math.radians(a1 + a2 + a3)
    gx, gy = x3, y3
    hand_len = 42
    spread = math.radians(22)

    left_tip = (
        gx + hand_len * math.cos(grip_angle + spread),
        gy + hand_len * math.sin(grip_angle + spread),
    )
    right_tip = (
        gx + hand_len * math.cos(grip_angle - spread),
        gy + hand_len * math.sin(grip_angle - spread),
    )

    # Gripper base
    pygame.draw.circle(surface, base_color, (int(gx), int(gy)), 16)
    pygame.draw.circle(surface, joint_color, (int(gx), int(gy)), 8)

    # Gripper fingers
    pygame.draw.line(surface, gripper_color, (gx, gy), left_tip, 8)
    pygame.draw.line(surface, gripper_color, (gx, gy), right_tip, 8)

    # Gripper tips
    pygame.draw.circle(surface, highlight, (int(left_tip[0]), int(left_tip[1])), 6)
    pygame.draw.circle(surface, highlight, (int(right_tip[0]), int(right_tip[1])), 6)

    return {
        "base": (x0, y0),
        "j1": (x1, y1),
        "j2": (x2, y2),
        "end": (x3, y3),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    pygame.init()
    pygame.display.set_caption("Gesture Arm Pro | AI Hand-Controlled Robotic Arm Simulator")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font_big = pygame.font.SysFont("Segoe UI", 30, bold=True)
    font_med = pygame.font.SysFont("Segoe UI", 22)
    font_small = pygame.font.SysFont("Segoe UI", 18)
    font_tiny = pygame.font.SysFont("Segoe UI", 15)

    # MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hand_style = mp_draw.DrawingSpec(color=(80, 220, 255), thickness=2, circle_radius=2)
    conn_style = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.72,
        min_tracking_confidence=0.72,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # Arm state
    base_x = CAM_W // 2
    base_y = HEIGHT - 120

    joint1_angle = -58.0
    joint2_angle = 52.0
    grip_angle = -18.0

    target_j1 = joint1_angle
    target_j2 = joint2_angle
    target_grip = grip_angle

    last_count = 0
    count_history = deque(maxlen=7)
    gesture_label = "..."
    action_label = "Waiting for hand"
    status_label = "Ready"
    status_color = GREEN

    motion_speed = 0.11
    cooldown = 0.0
    last_action_time = 0.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        cooldown = max(0.0, cooldown - dt)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # ------------------------
        # CAMERA + GESTURE
        # ------------------------
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            finger_count = 0
            handedness_label = "Right"
            hand_detected = False

            if results.multi_hand_landmarks:
                hand_detected = True
                hand = results.multi_hand_landmarks[0]
                landmarks = hand.landmark

                if results.multi_handedness:
                    handedness_label = results.multi_handedness[0].classification[0].label

                finger_count = count_fingers(hand, handedness_label)
                count_history.append(finger_count)
                stable_count = max(set(count_history), key=count_history.count)
                last_count = stable_count

                gesture_label, action_label = gesture_from_count(stable_count)

                # Draw hand overlay on camera frame
                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    hand_style,
                    conn_style,
                )

                # Map gestures to arm targets with smooth motion
                now = time.time()
                if cooldown == 0.0 and now - last_action_time > 0.08:
                    if stable_count == 1:
                        target_j1 -= 3.0
                        status_label = "Turning base left"
                        status_color = ACCENT
                    elif stable_count == 2:
                        target_j1 += 3.0
                        status_label = "Turning base right"
                        status_color = ACCENT
                    elif stable_count == 3:
                        target_j2 = clamp(target_j2 - 3.0, -35, 95)
                        status_label = "Lifting arm"
                        status_color = GREEN
                    elif stable_count == 4:
                        target_j2 = clamp(target_j2 + 3.0, -35, 95)
                        status_label = "Lowering arm"
                        status_color = ORANGE
                    elif stable_count == 5:
                        target_j1 = -58.0
                        target_j2 = 52.0
                        target_grip = -18.0
                        status_label = "Resetting pose"
                        status_color = YELLOW
                    else:
                        status_label = "Holding position"
                        status_color = MUTED

                    last_action_time = now
                    cooldown = 0.03
            else:
                count_history.clear()
                gesture_label = "NO HAND"
                action_label = "Show your hand to control the arm"
                status_label = "Searching for hand"
                status_color = RED

            cv2.putText(frame, f"Fingers: {last_count}", (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            cv2.putText(frame, f"Gesture: {gesture_label}", (18, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Action: {action_label}", (18, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 90), 2)
            cv2.imshow("Camera Feed", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                running = False

        # Smoothly move the arm toward targets
        joint1_angle = lerp(joint1_angle, target_j1, motion_speed)
        joint2_angle = lerp(joint2_angle, target_j2, motion_speed)
        grip_angle = lerp(grip_angle, target_grip, 0.08)

        # Some tiny idle bias for gripper realism
        if last_count == 0:
            target_grip = lerp(target_grip, -18.0, 0.05)

        # Clamp to readable ranges
        joint1_angle = clamp(joint1_angle, -135, 35)
        joint2_angle = clamp(joint2_angle, -40, 120)
        grip_angle = clamp(grip_angle, -45, 20)

        # ------------------------
        # DRAW UI BACKGROUND
        # ------------------------
        screen.fill(BG)

        # Canvas area with a subtle gradient style
        pygame.draw.rect(screen, BG_2, (0, 0, CAM_W, HEIGHT))
        pygame.draw.rect(screen, PANEL, (CAM_W, 0, PANEL_W, HEIGHT))

        # Left stage card
        stage_rect = pygame.Rect(28, 28, CAM_W - 56, HEIGHT - 56)
        draw_round_rect(screen, (17, 22, 36), stage_rect, radius=26)
        pygame.draw.rect(screen, (35, 45, 70), stage_rect, width=1, border_radius=26)

        # Right info panel cards
        card1 = pygame.Rect(CAM_W + 22, 24, PANEL_W - 44, 168)
        card2 = pygame.Rect(CAM_W + 22, 208, PANEL_W - 44, 222)
        card3 = pygame.Rect(CAM_W + 22, 446, PANEL_W - 44, 300)
        for c in (card1, card2, card3):
            draw_round_rect(screen, CARD, c, radius=22)
            pygame.draw.rect(screen, (48, 59, 89), c, width=1, border_radius=22)

        # Header
        draw_shadowed_text(screen, "Gesture Arm Pro", font_big, TEXT, (CAM_W + 34, 36))
        draw_shadowed_text(screen, "AI Hand-Controlled Robotic Arm Simulator", font_tiny, MUTED, (CAM_W + 34, 76))

        # Status pill
        pill = pygame.Rect(CAM_W + 34, 116, 140, 32)
        draw_round_rect(screen, (28, 55, 48) if status_color == GREEN else (60, 40, 30), pill, radius=16)
        draw_shadowed_text(screen, status_label, font_tiny, WHITE, (CAM_W + 48, 123), shadow_offset=(1, 1))

        # Arm metrics
        draw_shadowed_text(screen, f"Base Angle: {joint1_angle:6.1f}°", font_med, TEXT, (CAM_W + 34, 230))
        draw_shadowed_text(screen, f"Elbow Angle: {joint2_angle:6.1f}°", font_med, TEXT, (CAM_W + 34, 264))
        draw_shadowed_text(screen, f"Gripper:    {grip_angle:6.1f}°", font_med, TEXT, (CAM_W + 34, 298))

        # Gesture label card
        label_rect = pygame.Rect(CAM_W + 34, 352, PANEL_W - 68, 52)
        draw_round_rect(screen, (35, 47, 73), label_rect, radius=16)
        draw_shadowed_text(screen, f"Gesture: {gesture_label}", font_med, ACCENT_2, (CAM_W + 48, 364))

        # Control legend
        draw_shadowed_text(screen, "Controls", font_med, TEXT, (CAM_W + 34, 470))
        legend = [
            ("1 finger", "Rotate base left"),
            ("2 fingers", "Rotate base right"),
            ("3 fingers", "Lift arm"),
            ("4 fingers", "Lower arm"),
            ("5 fingers", "Reset pose"),
        ]
        y = 510
        for left, right in legend:
            draw_shadowed_text(screen, left, font_small, ACCENT, (CAM_W + 34, y))
            draw_shadowed_text(screen, right, font_small, MUTED, (CAM_W + 130, y))
            y += 34

        # Footer note
        footer = pygame.Rect(CAM_W + 22, HEIGHT - 78, PANEL_W - 44, 54)
        draw_round_rect(screen, (24, 31, 50), footer, radius=18)
        draw_shadowed_text(screen, "Press ESC to exit", font_tiny, MUTED, (CAM_W + 34, HEIGHT - 58))

        # ------------------------
        # DRAW ARM STAGE
        # ------------------------
        # Soft grid for depth
        for x in range(60, CAM_W - 40, 100):
            pygame.draw.line(screen, (23, 30, 48), (x, 60), (x, HEIGHT - 60), 1)
        for y in range(80, HEIGHT - 40, 100):
            pygame.draw.line(screen, (23, 30, 48), (60, y), (CAM_W - 60, y), 1)

        # Base platform glow
        pygame.draw.circle(screen, (30, 48, 70), (int(base_x), int(base_y + 18)), 68)
        pygame.draw.circle(screen, (27, 35, 56), (int(base_x), int(base_y + 18)), 58)

        # Robot arm
        joints = draw_robot_arm(screen, (base_x, base_y), joint1_angle, joint2_angle)

        # Small end-effector target marker
        ex, ey = joints["end"]
        pygame.draw.circle(screen, (255, 255, 255), (int(ex), int(ey)), 4)
        pygame.draw.circle(screen, (76, 201, 240), (int(ex), int(ey)), 10, width=2)

        # On-canvas instruction bubble
        bubble = pygame.Rect(60, HEIGHT - 112, 360, 48)
        draw_round_rect(screen, (21, 26, 43), bubble, radius=18)
        draw_shadowed_text(screen, "Show your hand to control the arm", font_tiny, MUTED, (76, HEIGHT - 96))

        pygame.display.flip()

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()
