import cv2

# Read the image
img = cv2.imread('Screenshot (90).png', 1)

# Draw a line on the image

# Get user input for text, font size, position, and color
text = input("Enter the text: ")
font_size = float(input("Enter the font size: "))
position_x = int(input("Enter the X-coordinate for text position: "))
position_y = int(input("Enter the Y-coordinate for text position: "))
text_color = tuple(map(int, input("Enter the text color (comma-separated BGR values): ").split(',')))

img1 = cv2.line(img, (50, 0), (200, 200), (0, 255, 0), 5)
# Add user-defined text to the image
font = cv2.FONT_HERSHEY_SIMPLEX
font_thickness = 2

cv2.putText(img1, text, (position_x, position_y), font, font_size, text_color, font_thickness)

# Display the image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
