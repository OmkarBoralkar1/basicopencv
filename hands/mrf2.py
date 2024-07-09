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

# Filter options
filter_options = {
    "grayscale": lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
    "blur": lambda img: cv2.GaussianBlur(img, (15, 15), 0),
    "negative": lambda img: cv2.bitwise_not(img)
}

# Filter UI elements
filter_rectangles = {
    "grayscale": (10, 30, 100, 50),
    "blur": (120, 30, 200, 50),
    "negative": (230, 30, 310, 50)
}

filter_selected = None  # Variable to keep track of the selected filter
filters_drawn = False  # Flag to check if filters have been drawn

def move_screen_content(direction):
    global frame_x, frame_y
    if direction == "left":
        frame_x -= 40
    elif direction == "right":
        frame_x += 40
    elif direction == "up":
        frame_y -= 40
    elif direction == "down":
        frame_y += 40

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

def select_filter(event, x, y, flags, param):
    global filter_selected, filters_drawn
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the click was within any of the filter areas
        for filter_name, rect in filter_rectangles.items():
            if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                filter_selected = filter_name
                filters_drawn = False  # Allow drawing filters again after selection
                break

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    # Create a larger background canvas
    canvas_height = frame.shape[0] * 3
    canvas_width = frame.shape[1] * 3
    background = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Calculate start indices for placing the frame within the background
    start_y = canvas_height // 2 - frame.shape[0] // 2 + frame_y
    start_x = canvas_width // 2 - frame.shape[1] // 2 + frame_x

    # Adjust start indices to ensure they are within bounds
    start_y = max(0, start_y)
    start_x = max(0, start_x)

    # Calculate end indices for placing the frame
    end_y = start_y+ frame.shape[0]
    end_x = start_x + frame.shape[1]

    # Ensure the end indices are within bounds
    end_y = min(end_y, canvas_height)
    end_x = min(end_x, canvas_width)

    # Place the frame within the background
    background[start_y:end_y, start_x:end_x] = frame[:end_y-start_y, :end_x-start_x]

    # Convert background frame to RGB color space for mediapipe
    rgb_frame = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    
    # Process the frame with Mediapipe
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(background, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the x and y coordinates of the index finger tip
            x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * canvas_width
            y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * canvas_height
            
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
                
                # Calculate rotation angle based on horizontal movement
                angle = get_angle([prev_x, prev_y], [x, y])
                rotated_frame = rotate_image(background, angle)
                background = rotated_frame  # Use the rotated frame for further processing

            # Update the previous hand position
            prev_x = x
            prev_y = y

    # Draw filter options within the frame
    if not filters_drawn:
        for name, rect in filter_rectangles.items():
            cv2.rectangle(background, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 0), -1)
            cv2.putText(background, name, (rect[0] + 10, rect[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        filters_drawn = True  # Prevent drawing filters again until a new selection is made

    # If a filter has been selected, apply it to the frame
    if filter_selected:
        frame_to_filter = background[start_y:end_y, start_x:end_x]
        filtered_frame = filter_options[filter_selected](frame_to_filter)
        background[start_y:end_y, start_x:end_x] = filtered_frame
        filter_selected = None  # Reset the selected filter

    # Display the resulting frame
    cv2.imshow('Frame', background)

    # Set mouse callback to handle filter selection
    cv2.setMouseCallback('Frame', select_filter)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()