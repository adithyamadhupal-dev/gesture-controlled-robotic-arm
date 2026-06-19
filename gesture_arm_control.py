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


def _jacobian_numeric(base, l1, l2, l3, a1, a2, delta_deg=1.0):
    """Numerical Jacobian of end-effector (x,y) w.r.t a1 and a2 (degrees -> radians handled).

    Returns J as [[dx/da1, dx/da2], [dy/da1, dy/da2]] where derivatives are per radian.
    """
    p0 = arm_forward_kinematics(base, l1, l2, l3, a1, a2, -a2 * 0.55)[2]
    delta_rad = math.radians(delta_deg)

    p_a1 = arm_forward_kinematics(base, l1, l2, l3, a1 + delta_deg, a2, -(a2) * 0.55)[2]
    p_a2 = arm_forward_kinematics(base, l1, l2, l3, a1, a2 + delta_deg, -(a2 + delta_deg) * 0.55)[2]

    j11 = (p_a1[0] - p0[0]) / delta_rad
    j21 = (p_a1[1] - p0[1]) / delta_rad

    j12 = (p_a2[0] - p0[0]) / delta_rad
    j22 = (p_a2[1] - p0[1]) / delta_rad

    return [[j11, j12], [j21, j22]]


def _ik_step_towards(base, l1, l2, l3, joint1_angle, joint2_angle, target_x, target_y, cur_tj1, cur_tj2, gain=0.6):
    """Apply one Jacobian-transpose IK step toward target (returns new target_j1, target_j2, end_x, end_y).

    - Angles are in degrees. gain scales the JT*error step (tuned small to avoid oscillation).
    - Uses numeric Jacobian with a small delta.
    """
    # Current end-effector position
    _, _, end = arm_forward_kinematics(base, l1, l2, l3, joint1_angle, joint2_angle, -joint2_angle * 0.55)
    end_x, end_y = end

    J = _jacobian_numeric(base, l1, l2, l3, joint1_angle, joint2_angle, delta_deg=1.0)

    # J = [[dx/da1, dx/da2], [dy/da1, dy/da2]] (per radian)
    # JT columns
    jt_col0 = (J[0][0], J[1][0])
    jt_col1 = (J[0][1], J[1][1])

    ex = target_x - end_x
    ey = target_y - end_y

    # delta_theta (radians) = gain * JT * error (pixels)
    dtheta0 = gain * (jt_col0[0] * ex + jt_col0[1] * ey)
    dtheta1 = gain * (jt_col1[0] * ex + jt_col1[1] * ey)

    # Convert to degrees for target_j updates
    ddeg0 = math.degrees(dtheta0)
    ddeg1 = math.degrees(dtheta1)

    new_tj1 = clamp(cur_tj1 + ddeg0, -135, 35)
    new_tj2 = clamp(cur_tj2 + ddeg1, -40, 120)

    return new_tj1, new_tj2, end_x, end_y


def draw_robot_arm(surface, base, joint1, joint2, gripper_open=26):
    # Realistic industrial-ish arm colors
    base_color = (52, 67, 88)
    arm_dark = (88, 120, 170)
    arm_light = (132, 183, 255)
    joint_color = (238, 240, 245)
    gripper_color = (210, 220, 235)
    highlight = (255, 255, 255)

    l1, l2, l3 = 190, 160, 85
    l1, l2, l3 = 230, 190, 100
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
    spread = math.radians(abs(gripper_open))

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

    ### Better Box Placement
    boxes = [
        {"x": 320, "y": 380, "size": 40, "attached": False},
        {"x": 420, "y": 330, "size": 40, "attached": False},
        {"x": 520, "y": 280, "size": 40, "attached": False},
    ]
    held_box = None
    robot_mode = "MANUAL"

    auto_state = "IDLE"

    auto_box = None

    score = 0

    drop_zone = {
        "x": 500,
        "y": 220,
        "w": 120,
        "h": 120
    }

    home_pose = (-58, 52)

    open_palm_start = None

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
                if robot_mode == "MANUAL" and cooldown == 0.0 and now - last_action_time > 0.08:
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
                        # Require a sustained open palm to activate autonomous mode
                        if open_palm_start is None:
                            open_palm_start = time.time()
                        elif time.time() - open_palm_start > 2.0:
                            if robot_mode != "AUTO":
                                robot_mode = "AUTO"
                                auto_state = "SELECT_BOX"
                                auto_box = None
                                print("AUTO STARTED")

                    elif stable_count == 0:

                        target_grip = -45
                        if held_box is None:

                            _, _, end_eff = arm_forward_kinematics(
                                (base_x, base_y),
                                230, 190, 100,
                                joint1_angle,
                                joint2_angle,
                                -joint2_angle * 0.55
                            )
                            end_x, end_y = end_eff

                            for box in boxes:

                                box_center_x = box["x"] + box["size"] / 2
                                box_center_y = box["y"] + box["size"] / 2

                                distance = math.hypot(
                                    end_x - box_center_x,
                                    end_y - box_center_y
                                )

                                if distance < 60:

                                    box["attached"] = True
                                    held_box = box
                                    break

                        status_label = "Closing gripper"
                        status_color = ORANGE

                    else:

                        status_label = "Holding position"
                        status_color = MUTED

                    # reset open-palm timer if user is no longer showing open palm
                    if stable_count != 5:
                        open_palm_start = None

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
        # ==================================================
        # AUTONOMOUS CONTROLLER
        # ==================================================

        print(
            "MODE =", robot_mode,
            "| STATE =", auto_state,
            "| AUTO_BOX =", auto_box is not None
        )

        # Compute the current end effector position before autonomous control.
        _, _, end_pos = arm_forward_kinematics(
            (base_x, base_y),
            230, 190, 100,
            joint1_angle,
            joint2_angle,
            -joint2_angle * 0.55
        )
        end_x, end_y = end_pos

        # Autonomous state-machine controller using Jacobian-transpose IK steps
        if robot_mode == "AUTO":

            if auto_state == "SELECT_BOX":
                # choose nearest unattached box to end-effector
                best = None
                best_d = float("inf")
                for box in boxes:
                    if not box["attached"]:
                        bx = box["x"] + box["size"] / 2
                        by = box["y"] + box["size"] / 2
                        d = math.hypot(end_x - bx, end_y - by)
                        if d < best_d:
                            best = box
                            best_d = d
                if best is None:
                    # nothing to do
                    robot_mode = "MANUAL"
                    auto_state = "IDLE"
                    auto_box = None
                else:
                    auto_box = best
                    auto_state = "MOVE_TO_BOX"

            elif auto_state == "MOVE_TO_BOX" and auto_box is not None:
                tx = auto_box["x"] + auto_box["size"] / 2
                ty = auto_box["y"] + auto_box["size"] / 2

                # IK step to move both joints toward the target
                target_j1, target_j2, end_x, end_y = _ik_step_towards(
                    (base_x, base_y), 230, 190, 100,
                    joint1_angle, joint2_angle,
                    tx, ty,
                    target_j1, target_j2,
                    gain=0.8,
                )

                distance = math.hypot(end_x - tx, end_y - ty)
                print("DISTANCE =", int(distance))

                if distance < 45:
                    auto_state = "GRAB"

            elif auto_state == "GRAB":
                target_grip = -45
                if auto_box is not None:
                    auto_box["attached"] = True
                    held_box = auto_box
                auto_state = "MOVE_TO_DROP"

            elif auto_state == "MOVE_TO_DROP":
                tx = drop_zone["x"] + drop_zone["w"] / 2
                ty = drop_zone["y"] + drop_zone["h"] / 2

                target_j1, target_j2, end_x, end_y = _ik_step_towards(
                    (base_x, base_y), 230, 190, 100,
                    joint1_angle, joint2_angle,
                    tx, ty,
                    target_j1, target_j2,
                    gain=0.8,
                )
                distance = math.hypot(end_x - tx, end_y - ty)

                if distance < 50:
                    auto_state = "RELEASE"

            elif auto_state == "RELEASE":
                target_grip = 20
                if held_box is not None:
                    held_box["attached"] = False
                    # place centered inside drop zone
                    held_box["x"] = drop_zone["x"] + (drop_zone["w"] - held_box["size"]) / 2
                    held_box["y"] = drop_zone["y"] + (drop_zone["h"] - held_box["size"]) / 2
                    held_box = None
                    score += 1
                auto_state = "RETURN_HOME"

            elif auto_state == "RETURN_HOME":
                # Move targets toward home_joint pose smoothly
                target_j1 = lerp(target_j1, home_pose[0], 0.12)
                target_j2 = lerp(target_j2, home_pose[1], 0.12)

                if (
                    abs(joint1_angle - home_pose[0]) < 3
                    and
                    abs(joint2_angle - home_pose[1]) < 3
                ):
                    auto_state = "IDLE"
                    robot_mode = "MANUAL"
                    auto_box = None

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
        mode_text = font_med.render(
            f"Mode: {robot_mode}",
            True,
            WHITE
        )

        screen.blit(
            mode_text,
            (CAM_W + 35, 410)
        )
        # Control legend
        score_text = font_med.render(
            f"Tasks Completed: {score}",
            True,
            WHITE
        )

        screen.blit(
            score_text,
            (CAM_W + 35, 440)
        )

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
        joints = draw_robot_arm(screen,(base_x, base_y),joint1_angle,joint2_angle,grip_angle)
        end_x, end_y = joints["end"]
        if held_box is not None and held_box["attached"]:

            held_box["x"] = end_x - held_box["size"] / 2
            held_box["y"] = end_y - held_box["size"] / 2
        # Small end-effector target marker
        ex, ey = joints["end"]
        pygame.draw.circle(screen, (255, 255, 255), (int(ex), int(ey)), 4)
        pygame.draw.circle(screen, (76, 201, 240), (int(ex), int(ey)), 10, width=2)

        # On-canvas instruction bubble
        bubble = pygame.Rect(60, HEIGHT - 112, 360, 48)
        draw_round_rect(screen, (21, 26, 43), bubble, radius=18)
        draw_shadowed_text(screen, "Show your hand to control the arm", font_tiny, MUTED, (76, HEIGHT - 96))
        pygame.draw.rect(
            screen,
            (70, 220, 120),
            (
                drop_zone["x"],
                drop_zone["y"],
                drop_zone["w"],
                drop_zone["h"]
            ),
            border_radius=12
        )

        label = font_small.render(
            "DROP ZONE",
            True,
            (255,255,255)
        )

        screen.blit(
            label,
            (
                drop_zone["x"] + 12,
                drop_zone["y"] + 18
            )
        )
        for box in boxes:

            box_center_x = box["x"] + box["size"] / 2
            box_center_y = box["y"] + box["size"] / 2

            distance = math.hypot(
                end_x - box_center_x,
                end_y - box_center_y
            )

            color = (255, 180, 50)
            if box["attached"]:
                color = (80, 200, 255)

            if distance < 60:
                color = (50, 255, 100)

            pygame.draw.rect(
                screen,
                color,
                (
                    box["x"],
                    box["y"],
                    box["size"],
                    box["size"]
                ),
                border_radius=6
            )

        pygame.display.flip()

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()
