import cv2
import os
import matplotlib.pyplot as plt  # Add this line

# Callback function for mouse events
def crop_image(event, x, y, flags, param):
    global drawing, top_left_pt, bottom_right_pt, img, img_resized, cropped_img_resized

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left_pt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)

        # Crop and show only the selected region
        if top_left_pt != (-1, -1) and bottom_right_pt != (-1, -1):
            cropped_img = img_resized[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]]
            cv2.imshow('Select Region', img_resized)

            # Resize the cropped image to the screen size
            cropped_img_resized = cv2.resize(cropped_img, (screen_width, screen_height))
            cv2.imshow('Cropped Image', cropped_img_resized)

            # Reset drawing area
            top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
            cv2.setMouseCallback('Select Region', crop_image)

            # Display the cropped image using Matplotlib
            titles = ['cropped_img']
            images = [cropped_img_resized]
            plt.imshow(cropped_img)
            plt.title(titles[0])  # Add title
            plt.show()  # Add parentheses

# Load the image
img = cv2.imread('Screenshot (90).png', 1)

# Check if the image is loaded successfully
if img is None:
    print("Error: Unable to load the image dwdtw.")
    exit()

# Get screen size
screen_width, screen_height = 1500, 700  # Adjust these values based on your screen size

# Resize the original image to the screen size
img_resized = cv2.resize(img, (screen_width, screen_height))

# Initialize global variables
drawing = False
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
cropped_img_resized = None

# Create a window and set the callback function
cv2.imshow('Select Region', img_resized)
cv2.setMouseCallback('Select Region', crop_image)

while True:
    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to quit
    if key == ord('q'):
        break

cv2.destroyAllWindows()
