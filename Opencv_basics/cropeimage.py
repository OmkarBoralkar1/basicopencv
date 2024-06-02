import cv2
import os
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Define img as a global variable
img = None

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
        if top_left_pt!= (-1, -1) and bottom_right_pt!= (-1, -1):
            cropped_img = img_resized[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]]
            cv2.imshow('Select Region', img_resized)

            # Resize the cropped image to the screen size
            cropped_img_resized = cv2.resize(cropped_img, (screen_width, screen_height))
            cv2.imshow('Cropped Image', cropped_img_resized)

            # Reset drawing area
            top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
            cv2.setMouseCallback('Select Region', crop_image)

            # Display the cropped image using Matplotlib
            titles = ['Cropped Image']
            images = [cropped_img_resized]
            plt.figure(figsize=(10, 10))  # Optional: Adjust figure size
            plt.imshow(cv2.cvtColor(cropped_img_resized, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
            plt.title(titles[0])  # Add title
            plt.axis('off')  # Hide axes
            plt.show()  # Show the plot

# Load the image
def select_image():
    print('select image')
    global img
    Tk().withdraw()  # Keep the root window from appearing
    
    filename = askopenfilename(filetypes=[("Image files", "*.jpeg;*.jpg;*.png")])
    
    if not filename:
        print("No file selected.")
        return None  # Return None if no file is selected
    
    # Check if the file exists
    if not os.path.isfile(filename):
        print(f"The file {filename} does not exist.")
        return None
    
    try:
        img = cv2.imread(filename)
        if img is None:
            raise ValueError(f"Unable to load the image {filename}.")
    except Exception as e:
        print(f"An error occurred while loading the image {filename}: {str(e)}")
        return None
    
    return img

img1=select_image()
# Check if the image is loaded successfully
if img1 is None:
    print("Error: Unable to load the image.")
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
cv2.namedWindow('Select Region')
cv2.setMouseCallback('Select Region', crop_image)

while True:
    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to quit
    if key == ord('q'):
        break

cv2.destroyAllWindows()
