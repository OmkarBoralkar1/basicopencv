import cv2
from matplotlib import pyplot as plt
# Read the original image
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

# Apply image processing operations
retval, threshold = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval2, threshold2 = cv2.threshold(grayscale, 255, 255, cv2.THRESH_BINARY)
gaue = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
laplacian_img = cv2.Laplacian(gaue, cv2.CV_64F)
colorized = cv2.applyColorMap(laplacian_img.astype('uint8'), cv2.COLORMAP_JET)
titles = ['Threshold Image', 'Adaptive Threshold Image', 'Laplacian Image', 'Colorized Image']
images = [threshold, gaue, laplacian_img, colorized]

# Display processed images using matplotlib
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])

plt.show()
# Display options for processed images
print("\nSelect an option for the processed image:")
print("  1. Threshold Image")
print("  2. Adaptive Threshold Image")
print("  3. Laplacian Image")
print("  4. Colorized Image")

choice_processed_img = input("Enter the number corresponding to the processed image you want to display: ")

# Display the selected processed image
if choice_processed_img == '1':
    img_processed = threshold
    img_processed_name = 'Threshold Image'
elif choice_processed_img == '2':
    img_processed = gaue
    img_processed_name = 'Adaptive Threshold Image'
elif choice_processed_img == '3':
    img_processed = laplacian_img
    img_processed_name = 'Laplacian Image'
elif choice_processed_img == '4':
    img_processed = colorized
    img_processed_name = 'Colorized Image'
else:
    print("Invalid choice for the processed image. Please enter a number from 1 to 4.")
    exit()

# Display the selected image and processed results
cv2.imshow(img_name, img)
cv2.imshow(img_processed_name, img_processed)
cv2.waitKey(0)
cv2.destroyAllWindows()
