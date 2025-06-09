import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
canvas = None

# Brush settings
black = (0, 0, 0, 255)
red = (0, 0, 255, 255)
eraser_color = (0, 0, 0, 0)  # transparent
draw_color = black
brush_thickness = 5
xp, yp = 0, 0

# Thickness control bar settings
bar_height = 50
bar_color = (200, 200, 200)
bar_range = (10, 100)
bar_start = 100
bar_end = 500

def get_finger_status(lm):
    """Returns dict of open fingers"""
    return {
        'thumb': lm[4].x > lm[3].x,
        'index': lm[8].y < lm[6].y,
        'middle': lm[12].y < lm[10].y,
        'ring': lm[16].y < lm[14].y,
        'pinky': lm[20].y < lm[18].y,
    }

def count_open(status):
    return sum(status.values())

def draw_thickness_bar(frame, thickness):
    """Draws horizontal brush thickness control bar"""
    cv2.rectangle(frame, (bar_start, 10), (bar_end, 10 + bar_height), bar_color, 2)
    marker_x = int(np.interp(thickness, bar_range, [bar_start, bar_end]))
    cv2.circle(frame, (marker_x, 10 + bar_height // 2), 10, (0, 0, 255), -1)
    cv2.putText(frame, f"Thickness: {thickness}", (bar_start, bar_height + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 4), dtype=np.uint8)  # Transparent canvas

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Draw thickness bar
    draw_thickness_bar(frame, brush_thickness)

    if results.multi_hand_landmarks:
        all_hands = results.multi_hand_landmarks

        for i, hand in enumerate(all_hands):
            lm = hand.landmark
            fingers = get_finger_status(lm)

            index_x, index_y = int(lm[8].x * w), int(lm[8].y * h)

            # Clear gesture — all fingers open
            if i == 0 and count_open(fingers) == 5:
                canvas = np.zeros((h, w, 4), dtype=np.uint8)
                cv2.putText(frame, "Canvas Cleared", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                xp, yp = 0, 0
                continue

            # Pause if thumb is up
            if fingers['thumb']:
                xp, yp = 0, 0
                continue

            # Thickness control (index inside bar)
            if index_y < 10 + bar_height and bar_start <= index_x <= bar_end:
                brush_thickness = int(np.interp(index_x, [bar_start, bar_end], bar_range))
                continue

            # Drawing mode
            if fingers['index'] and fingers['middle']:
                draw_color = red
            elif fingers['index'] and not fingers['middle']:
                draw_color = black
            else:
                xp, yp = 0, 0
                continue

            if xp == 0 and yp == 0:
                xp, yp = index_x, index_y

            cv2.line(canvas, (xp, yp), (index_x, index_y), draw_color, brush_thickness)
            xp, yp = index_x, index_y

            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # Overlay canvas (transparent) onto frame
    overlay = frame.copy()
    alpha_canvas = canvas[:, :, 3] / 255.0
    for c in range(3):
        overlay[:, :, c] = overlay[:, :, c] * (1 - alpha_canvas) + canvas[:, :, c] * alpha_canvas

    cv2.imshow("Transparent Air Canvas", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
