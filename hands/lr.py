import cv2
import numpy as np
import mediapipe as mp

# Initialize webcam
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Variables to store previous hand position
prev_x = None
prev_y = None
frame_x = 0  # Initial x position of the frame within the canvas
frame_y = 0  # Initial y position of the frame within the canvas

def move_screen_content(direction):
    global frame_x, frame_y
    if direction == "left":
        frame_x -= 50
    elif direction == "right":
        frame_x += 50
    elif direction == "up":
        frame_y -= 50
    elif direction == "down":
        frame_y += 50

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB color space for mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with Mediapipe
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the x and y coordinates of the index finger tip
            x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]
            y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]
            
            # Determine the direction of movement
            if prev_x is not None and prev_y is not None:
                if x > prev_x + 20:
                    move_screen_content("right")
                elif x < prev_x - 20:
                    move_screen_content("left")
                if y > prev_y + 20:
                    move_screen_content("down")
                elif y < prev_y - 20:
                    move_screen_content("up")
            
            # Update the previous hand position
            prev_x = x
            prev_y = y

    # Create a larger canvas to move the frame within
    canvas_width = 3 * frame.shape[1]
    canvas_height = 3 * frame.shape[0]
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Ensure the frame doesn't move out of the canvas boundaries
    frame_x = np.clip(frame_x, 0, canvas_width - frame.shape[1])
    frame_y = np.clip(frame_y, 0, canvas_height - frame.shape[0])

    # Place the frame within the canvas at the specified position
    canvas[frame_y:frame_y + frame.shape[0], frame_x:frame_x + frame.shape[1]] = frame

    # Display the resulting canvas
    cv2.imshow('Frame', canvas)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()