import cv2
import numpy as np
import mediapipe as mp
import math

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

def get_angle(p1, p2):
    """Calculate the angle between two vectors."""
    v1 = np.array(p1)
    v2 = np.array(p2)
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)

def rotate_image(image, angle):
    """
    Rotates an image (assumed to be in BGR format) around its center by a given angle.
    :param image: The image to rotate.
    :param angle: The angle by which to rotate the image.
    :return: The rotated image.
    """
    (height, width) = image.shape[:2]
    (cX, cY) = (width // 2, height // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((height * sin) + (width * cos))
    nH = int((height * cos) + (width * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))

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
            
            # Calculate the angle between the vertical axis and the vector from the wrist to the index finger tip
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            angle = get_angle([wrist.x, wrist.y], [index_finger_tip.x, index_finger_tip.y])

            # Apply rotation to the frame
            rotated_frame = rotate_image(frame, angle)

            # Update the frame variable to use the rotated frame
            frame = rotated_frame
            
            # Update the previous hand position
            x = index_finger_tip.x
            y = index_finger_tip.y
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
