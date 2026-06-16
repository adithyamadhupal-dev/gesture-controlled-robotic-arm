import cv2
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = hand_landmarks.landmark

            # Count fingers (excluding thumb for now)
            finger_count = 0

            fingertip_ids = [8, 12, 16, 20]

            for tip in fingertip_ids:
                if landmarks[tip].y < landmarks[tip - 2].y:
                    finger_count += 1

            # Display count
            cv2.putText(
                frame,
                f"Fingers: {finger_count}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    # Show window
    cv2.imshow("Finger Counter", frame)

    # ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()