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

# Create a window named 'Frame'
cv2.namedWindow('Frame')
# Set the event callback for the window
cv2.setMouseCallback('Frame', select_filter)

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
                
                # Calculate rotation angle based on horizontal movement
                angle = get_angle([prev_x, prev_y], [x, y])
                rotated_frame = rotate_image(frame, angle)
                frame = rotated_frame  # Use the rotated frame for further processing

            # Update the previous hand position
            prev_x = x
            prev_y = y

    # Create a larger canvas to move the frame within
    canvas_width = 3 * frame.shape[1]
    canvas_height = 3 * frame.shape[0] + 100  # Add extra space for filter options
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # Create a white canvas

    # Ensure the frame doesn't move out of the canvas boundaries
    frame_x = np.clip(frame_x, 0, canvas_width - frame.shape[1])
    frame_y = np.clip(frame_y, 0, canvas_height - frame.shape[0] - 100)

    # If a filter has been selected, apply it to the frame
    if filter_selected:
        frame = filter_options[filter_selected](frame)
        filter_selected = None  # Reset the selected filter

    # Draw filter options within the canvas at the specified position
    if not filters_drawn:
        filter_area_width = canvas_width // len(filter_options)  # Calculate width for each filter area
        filter_area_height = 100  # Height for filter area

        for idx, (name, rect) in enumerate(filter_options.items()):
            print("the filters got are", name)
            print("the size of the filters got are", rect)
            x1 = idx * filter_area_width
            y1 = canvas_height - filter_area_height
            x2 = (idx + 1) * filter_area_width
            y2 = canvas_height
            print("the x1  got are", x1)
            print("the x2  got are", x2)
            print("the y1  got are", y1)
            print("the y2  got are", y2)
            # Draw rectangle with yellow background
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), -1)
            
            # Draw text label
            cv2.putText(canvas, name, (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        filters_drawn = True  # Set flag to true after drawing the filters
    
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
