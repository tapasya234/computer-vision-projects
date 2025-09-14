# Generates a live sketch of the frame pulled from the webcam

import cv2
import numpy


# Generates the sketch of the image passed in
def sketch(image):
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurredImg = cv2.GaussianBlur(imgGray, (5, 5), 0)
    edgesImg = cv2.Canny(blurredImg, 20, 60)
    _, mask = cv2.threshold(edgesImg, 70, 255, cv2.THRESH_BINARY_INV)
    return mask


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("Our Live Sketcher", sketch(frame))
    if cv2.waitKey(1) == 13:  # 13 represents the ENTER key
        break

cap.release()
cv2.destroyAllWindows()
