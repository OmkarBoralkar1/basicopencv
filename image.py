import cv2
from matplotlib import pyplot as plt

# Read the cropped image
original_img = cv2.imread('smoke picture.jpeg', 1)

# Display options and ask the user to choose between the original and resized image
print("Select an option:")
print("  1. Original Image")
print("  2. Resized Image")

choice_img = input("Enter the number corresponding to the image you want to process: ")

# Process the selected image
if choice_img == '1':
    img = original_img
    img_name = 'Original Image'
elif choice_img == '2':
    width =int(input("Enter the width of the resize image: "))
    height =int(input("Enter the height of the resize image: "))

    img = cv2.resize(original_img, (width, height))
    img_name = 'Resized Image'
else:
    print("Invalid choice for the image. Please enter '1' or '2'.")
    exit()
retval, threshold = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval2, threshold2 = cv2.threshold(grayscale, 255, 255, cv2.THRESH_BINARY)
gaue = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
laplacian_img = cv2.Laplacian(gaue, cv2.CV_64F)
colorized = cv2.applyColorMap(gaue.astype('uint8'), cv2.COLORMAP_JET)
thermal_frame = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)
titles = ['Threshold Image', 'Adaptive Threshold Image', 'Laplacian Image', 'Colorized Image']
images = [threshold, gaue, laplacian_img, colorized]

while True:
    # Display options
    print("Select an option:")
    print("1. Original Image")
    print("2. Grayscale Image")
    print("3. Blurred Image")
    print("4. Laplacian Image")
    print("5. Save Current Image")
    print("6. Threshold Image")
    print("7. Adaptive Threshold Image")
    print("8. Thermal")
    print("9. Colorized Image")
    print("10. Quit")

    choice = input("Enter your choice: ")

    if choice == '1':
        cv2.imshow('Image', img)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
    elif choice == '2':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.imshow('Grayscale Image', gray1)
        plt.imshow(cv2.cvtColor(gray1, cv2.COLOR_BGR2RGB))
        plt.show()
    elif choice == '3':
        blurred_img = cv2.GaussianBlur(img, (5, 5), 10000)
        cv2.imshow('Blurred Image', blurred_img)
        plt.imshow(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB))
        plt.show()
    elif choice == '4':
        cv2.imshow('Laplacian Image', laplacian_img)
        plt.imshow(cv2.cvtColor(laplacian_img.astype('uint8'), cv2.COLOR_BGR2RGB))
        plt.show()
    elif choice == '5':
        save_choice = input("Enter a filename to save the current image: ")
        cv2.imwrite(save_choice, img)
        print(f"Image saved as {save_choice}")
    elif choice == '6':
        img_processed = threshold
        img_processed_name = 'Threshold Image'
        cv2.imshow(img_processed_name, img_processed)
        plt.imshow(cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB))
        plt.show()
    elif choice == '7':
        img_processed = gaue
        img_processed_name = 'Adaptive Threshold Image'
        cv2.imshow(img_processed_name, img_processed)
        plt.imshow(cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB))
        plt.show()
    elif choice == '8':
        img_processed = thermal_frame
        img_processed_name = 'thermal_frame'
        cv2.imshow(img_processed_name, img_processed)
        plt.imshow(cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB))
        plt.show()
    elif choice == '9':
        img_processed = colorized
        img_processed_name = 'Colorized Image'
        cv2.imshow(img_processed_name, img_processed)
        plt.imshow(cv2.cvtColor(img_processed, cv2.COLOR_RGB2LUV))
        plt.show()
    elif choice == '10':
        break
    else:
        print("Invalid choice. Please enter a valid option.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
