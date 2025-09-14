import cv2
from data_path import DATA_PATH
import numpy as np
import glob
import sys

lowerGreen = np.array([40, 100, 70])
upperGreen = np.array([90, 255, 255])

windowName = "Output"


def updateVideo(videoName, background):
    cap = cv2.VideoCapture(DATA_PATH + "greenScreenInput/" + videoName)
    if not cap.isOpened():
        sys.exit()

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        cv2.imshow(windowName, getUpdatedImage(frame, background))
        if cv2.waitKey(1) == 13:
            break

    cap.release()


def updateSingleImage(image, background):
    cv2.imshow(windowName, getUpdatedImage(image, background))
    cv2.waitKey(0)


def updateMultipleImages(background):
    for path in glob.glob(DATA_PATH + "greenScreenInput/greenScreen*.jpg"):
        input = cv2.imread(path)
        cv2.imshow(windowName, getUpdatedImage(input, background))
        cv2.waitKey(0)


def getUpdatedImage(input, background):
    inputHSV = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    backgroundResized = cv2.resize(background, (input.shape[1], input.shape[0]))

    backgroundMask = cv2.inRange(inputHSV, lowerGreen, upperGreen)
    foregroundMask = cv2.bitwise_not(backgroundMask)
    updatedForeground = cv2.bitwise_and(input, input, mask=foregroundMask)
    updatedBackground = cv2.bitwise_and(
        backgroundResized, backgroundResized, mask=backgroundMask
    )
    return cv2.bitwise_xor(updatedBackground, updatedForeground)


background = cv2.imread(DATA_PATH + "background/background5.jpg")

# updateVideo("greenscreen-demo.mp4", background)
# updateVideo("greenscreen-asteroid.mp4", background)

input = cv2.imread(DATA_PATH + "greenScreenInput/greenScreenInput5.jpg")
updateSingleImage(input, background)

# updateMultipleImages(background)

cv2.destroyAllWindows()
