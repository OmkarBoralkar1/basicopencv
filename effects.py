import cv2
import threading
import os
import sys

def create_video_writer(effect_name, index, width, height):
    base_name = f'{effect_name}_output_{index}'
    index = 1
    file_name = f'{base_name}.avi'

    while os.path.exists(file_name):
        file_name = f'{base_name}_{index}.avi'
        index += 1

    return cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))

def apply_effects(video_path, width, height, output_prefix):
    webcam = cv2.VideoCapture(video_path)

    show_original_frame = False
    show_grayscale_frame = False
    show_laplacian = False
    show_sobel_x_gradient = False
    show_sobel_y_gradient = False
    Adaptive_Threshold_Image = False
    show_thermal_radiation = False

    writers = {}

    def get_user_choice():
        nonlocal show_original_frame, show_grayscale_frame, show_laplacian, show_sobel_x_gradient, show_sobel_y_gradient, Adaptive_Threshold_Image, show_thermal_radiation

        while True:
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
                    writers['original'] = create_video_writer('original', output_prefix, width, height)
            elif choice == '2':
                show_grayscale_frame = not show_grayscale_frame
                if show_grayscale_frame:
                    writers['grayscale'] = create_video_writer('grayscale', output_prefix, width, height)
            elif choice == '3':
                show_laplacian = not show_laplacian
                if show_laplacian:
                    writers['laplacian'] = create_video_writer('laplacian', output_prefix, width, height)
            elif choice == '4':
                show_sobel_x_gradient = not show_sobel_x_gradient
                if show_sobel_x_gradient:
                    writers['sobel_x'] = create_video_writer('sobel_x', output_prefix, width, height)
            elif choice == '5':
                show_sobel_y_gradient = not show_sobel_y_gradient
                if show_sobel_y_gradient:
                    writers['sobel_y'] = create_video_writer('sobel_y', output_prefix, width, height)
            elif choice == '6':
                Adaptive_Threshold_Image = not Adaptive_Threshold_Image
                if Adaptive_Threshold_Image:
                    writers['adaptive_threshold'] = create_video_writer('adaptive_threshold', output_prefix, width, height)
            elif choice == '7':
                show_thermal_radiation = not show_thermal_radiation
                if show_thermal_radiation:
                    writers['thermal_radiation'] = create_video_writer('thermal_radiation', output_prefix, width, height)
            else:
                print("Invalid choice. Please enter a valid option.")

    # Start the user input thread
    user_input_thread = threading.Thread(target=get_user_choice)
    user_input_thread.start()

    while True:
        ret, frame = webcam.read()

        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display windows based on the flags
        if show_original_frame:
            cv2.imshow(f'Original Frame {output_prefix}', frame)
            writers['original'].write(frame)

        if show_grayscale_frame:
            cv2.imshow(f'Grayscale Frame {output_prefix}', gray_frame)
            writers['grayscale'].write(cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR))

        if show_laplacian:
            laplacian_frame = cv2.Laplacian(frame, cv2.CV_64F)
            cv2.imshow(f'Laplacian {output_prefix}', laplacian_frame)
            writers['laplacian'].write(laplacian_frame)

        if show_sobel_x_gradient:
            sobel_x_frame = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
            cv2.imshow(f'Sobel X-Gradient {output_prefix}', sobel_x_frame)
            writers['sobel_x'].write(sobel_x_frame)

        if show_sobel_y_gradient:
            sobel_y_frame = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
            cv2.imshow(f'Sobel Y-Gradient {output_prefix}', sobel_y_frame)
            writers['sobel_y'].write(sobel_y_frame)

        if Adaptive_Threshold_Image:
            adaptive_threshold_frame = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                             cv2.THRESH_BINARY, 115, 1)
            cv2.imshow('Adaptive_Threshold_Image', adaptive_threshold_frame)
            writers['adaptive_threshold'].write(cv2.cvtColor(adaptive_threshold_frame, cv2.COLOR_GRAY2BGR))

        if show_thermal_radiation:
            thermal_frame = cv2.applyColorMap(gray_frame, cv2.COLORMAP_JET)
            cv2.imshow(f'Thermal Radiation {output_prefix}', thermal_frame)
            writers['thermal_radiation'].write(thermal_frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # Release the VideoWriter objects
    for writer in writers.values():
        writer.release()

    # Wait for the user input thread to finish
    user_input_thread.join()

    webcam.release()
    cv2.destroyAllWindows()
    if __name__ == "__main__":
        if len(sys.argv) < 3:
            print("Usage: python apply_effects.py <video_path> <output_prefix>")
            sys.exit(1)

        video_path = sys.argv[1]
        output_prefix = sys.argv[2]

        print(f"Selected video: {video_path}")

        print("1. Keep the original video length")
        print("2. Resize the video with respect to its width and height")
        choice = int(input("Enter the number corresponding to your choice: "))

        if choice == 1:
            width, height = 0, 0  # Original video length
        elif choice == 2:
            width = int(input("Enter the width of the resized screen: "))
            height = int(input("Enter the height of the resized screen: "))
        else:
            print("Invalid choice. Using the original video length.")
            width, height = 0, 0
