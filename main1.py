import cv2
from effects import apply_effects

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

if __name__ == "__main__":
    # Number of videos to run simultaneously
    r = int(input('Enter the number of videos you want to run simultaneously: '))

    # List of video paths
    video_paths = [choose_video() for _ in range(r)]

    # Get user input for width and height
    width = int(input("Enter the width of the resized screen: "))
    height = int(input("Enter the height of the resized screen: "))

    # Output prefix (you might want to customize this)
    output_prefix = "output"

    print("Selected videos:")
    for i, video_path in enumerate(video_paths):
        print(f"{i + 1}. {video_path}")

    # Apply effects for each video
    for i, video_path in enumerate(video_paths):
        print('Applying effects to video:', video_path)
        apply_effects(video_path, width, height, i + 1, )
