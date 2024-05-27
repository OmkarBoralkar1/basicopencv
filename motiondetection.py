import cv2
import datetime

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
print(frame1.shape)  # Print the shape to check dimensions
ret, frame2 = cap.read()
print(frame2.shape)  # Print the shape to check dimensions

width = int(input("Enter the desired width: "))
height = int(input("Enter the desired height: "))

# Create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame1 = cap.read()
    if not ret:
        print("Error reading frame.")
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame1)

    # Further processing to enhance the result
    blur = cv2.GaussianBlur(fgmask, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Resize the frame
    frame1 = cv2.resize(frame1, (width, height))

    motion_detected = False  # Flag to indicate motion detection

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 700:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True

    if motion_detected:
        # Update the reference frame only if motion is detected
        frame2 = cap.read()[1]

    datet = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add date and time text to the frame
    cv2.putText(frame1, datet, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("motion", frame1)
    frame1 = frame2

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
