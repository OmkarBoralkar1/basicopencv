import cv2
from matplotlib import pyplot as plt

def remove_smoke(image):
    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to separate smoke from the background
    _, mask = cv2.threshold(grayscale, 150, 255, cv2.THRESH_BINARY)

    # Invert the mask to keep the background and remove the smoke
    result = cv2.bitwise_and(image, image, mask=~mask)

    return result

# Read the original image
img1 = cv2.imread('Screenshot (90).png', 1)
width = int(input("Enter the width of the resize image: "))
height = int(input("Enter the height of the resize image: "))

img = cv2.resize(img1, (width, height))
img_name = 'Resized Image'
# Remove smoke from the image
image_without_smoke = remove_smoke(img)

# Display the original and processed images using OpenCV
cv2.imshow('Original Image', img)
cv2.imshow('Image without Smoke', image_without_smoke)

# Display the processed image using matplotlib
plt.imshow(img)
plt.title('Image without Smoke')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
