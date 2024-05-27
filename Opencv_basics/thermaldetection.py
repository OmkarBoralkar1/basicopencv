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


# Modify the visualize_temperature function
def visualize_temperature(temperature_data):
    # Normalize the temperature data to 8-bit range (0-255)
    normalized_data = cv2.normalize(temperature_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Map temperature values to a color scale
    color_map = cv2.applyColorMap(normalized_data, cv2.COLORMAP_JET)

    # Display temperature information
    average_temperature = np.mean(temperature_data)
    # Display temperature information in both Celsius and Fahrenheit
    average_temperature_celsius = np.mean(average_temperature)
    average_temperature_fahrenheit = (average_temperature_celsius * 9 / 5) + 32

    temperature_info = (f"Average Temperature: {average_temperature_celsius:.2f} C "
                        f"/ {average_temperature_fahrenheit:.2f} F")
    cv2.putText(color_map, temperature_info, (-10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return color_map


# Modify the get_temperature_data function to return a uint8 array
def get_temperature_data(frame):
    # Replace this with your code to obtain temperature data from the thermal camera
    # The result should be a 2D array where each element represents the temperature value for a pixel
    temperature_data = np.random.uniform(25, 40, frame.shape[:2])  # Example random temperature data

    # Convert temperature data to uint8
    temperature_data = np.clip(temperature_data, 0, 255).astype(np.uint8)

    return temperature_data


# Start the user input thread
def get_user_choice():
    global show_original_frame, show_thermal_radiation

    while True:
        # Display options
        print("Select an option:")
        print("1. Toggle Original Frame")
        print("2. Toggle Thermal Radiation")
        choice = input("Enter your choice (q to quit): ")

        if choice == 'q':
            break
        elif choice == '1':
            show_original_frame = not show_original_frame
            if show_original_frame:
                writers['original'] = create_video_writer('original')
        elif choice == '2':
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

    # Replace grayscale frame with temperature data
    temperature_data = get_temperature_data(frame)

    # Display windows based on the flags
    if show_original_frame:
        cv2.imshow('Original Frame', frame)
        writers['original'].write(frame)

    if show_thermal_radiation:
        # Convert temperature data to 3 channels (RGB format) for applying color map
        temperature_colored = cv2.applyColorMap(cv2.cvtColor(temperature_data, cv2.COLOR_GRAY2BGR), cv2.COLORMAP_JET)
        thermal_frame = cv2.addWeighted(frame, 0.5, temperature_colored, 0.5, 0)

        thermal_frame_with_temperature = visualize_temperature(temperature_data)

        # Concatenate the frames horizontally for display
        display_frame = np.hstack([thermal_frame, thermal_frame_with_temperature])

        # Show the concatenated frame
        cv2.imshow('Thermal Radiation', display_frame)

        # Write both frames to the video file
        writers['thermal_radiation'].write(thermal_frame)
        writers['thermal_radiation'].write(thermal_frame_with_temperature)

    # ... (other display options)

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