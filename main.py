import datetime
import cv2
import threading
import numpy as np
import os

# Function to get user's choice for video source
def choose_video():
    print("Select a video source:")
    print("1. 'smoke video 2.mp4'")
    print("2. 'smoke video.mp4'")
    print("3. 'Ir sensor.mp4'")
    print("4. Enter custom video path (Webcam)")

    choice = input("Enter the number corresponding to your choice: ")

    if choice == '1':
        return 'smoke video 2.mp4'
    elif choice == '2':
        return 'smoke video.mp4'
    elif choice == '3':
        return 'Ir sensor.mp4'
    elif choice == '4':
        custom_choice = input("Enter custom video path (e.g., 0 for webcam): ")
        return custom_choice
    else:
        print("Invalid choice. Using default video path 'smoke video 2.mp4'.")
        return 'smoke video 2.mp4'

# Set the initial video path
video_path = choose_video()

# Open the video capture
if video_path.isdigit():
    # If the video path is a digit, assume it's the webcam number
    webcam1 = cv2.VideoCapture(int(video_path))
else:
    # Otherwise, treat it as a file path
    webcam1 = cv2.VideoCapture(video_path)

print("1. Keep the original video length")
print("2. Resize the video with respect to its width and height")
choice = int(input("Enter the number corresponding to your choice: "))

if choice == 1:
    webcam = webcam1
    width = int(webcam1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(webcam1.get(cv2.CAP_PROP_FRAME_HEIGHT))
elif choice == 2:
    width = int(input("Enter the width of the resized screen: "))
    height = int(input("Enter the height of the resized screen: "))
else:
    print("Invalid choice. Using the original video length.")
    webcam = webcam1

# Flags to control the display of each window
show_original_frame = False
show_grayscale_frame = False
show_laplacian = False
show_sobel_x_gradient = False
show_sobel_y_gradient = False
Adaptive_Threshold_Image = False
show_thermal_radiation = False

# Video writers for each effect
writers = {}

def create_video_writer(name):
    base_name = f"{name}.avi"
    index = 1

    while os.path.exists(base_name):
        base_name = f"{name}{index}.avi"
        index += 1

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(base_name, fourcc, 20.0, (width, height))

# Create a temperature scale for thermal visualization
temperature_scale = np.arange(0, 256, dtype=np.uint8).reshape(1, -1)

def visualize_temperature(image):
    # Map pixel values to a color scale (e.g., from blue to red)
    color_map = cv2.applyColorMap(image, cv2.COLORMAP_JET)

    # Convert pixel values to temperature values
    temperature_values = cv2.LUT(image, temperature_scale)

    # Display temperature information
    average_temperature = np.mean(temperature_values)
    temperature_info = f"Average Temperature: {average_temperature:.2f} C"
    cv2.putText(color_map, temperature_info, (-10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return color_map

def get_user_choice():
    global show_original_frame, show_grayscale_frame, show_laplacian, show_sobel_x_gradient, show_sobel_y_gradient, Adaptive_Threshold_Image, show_thermal_radiation

    while True:
        # Display options
        print("Select an option:")
        print("1. Original Frame")
        print("2. Grayscale Frame")
        print("3. Laplacian")
        print("4. Sobel X-Gradient")
        print("5. Sobel Y-Gradient")
        print("6. Adaptive Threshold Image")
        print("7. Thermal Radiation")
        choice = input("Enter your choice (q to quit): ")

        if choice == 'q':
            break
        elif choice == '1':
            show_original_frame = not show_original_frame
            if show_original_frame:
                writers['original'] = create_video_writer('original')
        elif choice == '2':
            show_grayscale_frame = not show_grayscale_frame
            if show_grayscale_frame:
                writers['grayscale'] = create_video_writer('grayscale')
        elif choice == '3':
            show_laplacian = not show_laplacian
            if show_laplacian:
                writers['laplacian'] = create_video_writer('laplacian')
        elif choice == '4':
            show_sobel_x_gradient = not show_sobel_x_gradient
            if show_sobel_x_gradient:
                writers['sobel_x'] = create_video_writer('sobel_x')
        elif choice == '5':
            show_sobel_y_gradient = not show_sobel_y_gradient
            if show_sobel_y_gradient:
                writers['sobel_y'] = create_video_writer('sobel_y')
        elif choice == '6':
            Adaptive_Threshold_Image = not Adaptive_Threshold_Image
            if Adaptive_Threshold_Image:
                writers['adaptive_threshold'] = create_video_writer('adaptive_threshold')
        elif choice == '7':
            show_thermal_radiation = not show_thermal_radiation
            if show_thermal_radiation:
                writers['thermal_radiation'] = create_video_writer('thermal_radiation')
        else:
            print("Invalid choice. Please enter a valid option.")

# Start the user input thread
user_input_thread = threading.Thread(target=get_user_choice)
user_input_thread.start()

while True:
    ret, frame = webcam1.read()

    if not ret:
        # Break the loop if the video has ended
        break

    datet = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, datet, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    if choice == 2:
        frame = cv2.resize(frame, (width, height))  # Resize the frame if the user chose to resize

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display windows based on the flags
    if show_original_frame:
        cv2.imshow('Original Frame', frame)
        writers['original'].write(frame)

    if show_grayscale_frame:
        cv2.imshow('Grayscale Frame', gray)
        writers['grayscale'].write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    if show_laplacian:
        laplacian_frame = cv2.Laplacian(frame, cv2.CV_64F)
        cv2.imshow('Laplacian', laplacian_frame)
        writers['laplacian'].write(laplacian_frame)

    if show_sobel_x_gradient:
        sobel_x_frame = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
        cv2.imshow('Sobel X-Gradient', sobel_x_frame)
        writers['sobel_x'].write(sobel_x_frame)

    if show_sobel_y_gradient:
        sobel_y_frame = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
        cv2.imshow('Sobel Y-Gradient', sobel_y_frame)
        writers['sobel_y'].write(sobel_y_frame)

    if Adaptive_Threshold_Image:
        adaptive_threshold_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                         115, 1)
        cv2.imshow('Adaptive_Threshold_Image', adaptive_threshold_frame)
        writers['adaptive_threshold'].write(cv2.cvtColor(adaptive_threshold_frame, cv2.COLOR_GRAY2BGR))

    if show_thermal_radiation:
        thermal_frame = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        thermal_frame_with_temperature = visualize_temperature(gray)
        cv2.imshow('Thermal Radiation', np.hstack([thermal_frame, thermal_frame_with_temperature]))
        writers['thermal_radiation'].write(thermal_frame)

    if cv2.waitKey(20) & 0xFF == 27:  # ASCII value for the Esc key
        break

# Wait for the user input thread to finish
user_input_thread.join()

# Release the VideoWriter objects
for writer in writers.values():
    writer.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Release the webcam
webcam1.release()
