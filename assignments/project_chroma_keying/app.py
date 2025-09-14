# In this project, you will implement any algorithm of your choice for chroma-keying. We have linked to a few resources and pieces of code you can look at.
# Input: The input to the algorithm will be a video with a subject in front of a green screen.
# Output: The output should be another video where the green background is replaced with an interesting background of your choice.
# The new background could even be a video if you want to make it interesting.
# Controls: You can build a simple interface using HighGUI. It should contain the following parts.
#  - Color Patch Selector : The interface should show a video frame and the user should be allowed to select
#    a patch of green screen from the background. For simplicity, this patch can be a rectangular patch selected from a single frame.
#    However, it is perfectly fine to build an interface where you select multiple patches from one or more frames of the video.
#  - Tolerance slider : This slider will control how different the color can be from the mean of colors sampled
#    in the previous step to be included in the green background.
#  - Softness slider (Optional): This slider will control the softness of the foreground mask at the edges.
#  - Color cast slider (Optional): In a green screen environment, some of the green color also gets cast on the subject.
#    There are some interesting ways the color cast can be reduced, but during the process of removing the color cast some artifacts are get introduced.
#    So, this slider should control the amount of color cast removal we want.

from functools import partial
import cv2
from data_path import DATA_PATH
import sys
import numpy as np
import glob

# Window names
outputWindowName = "Output Video"
inputWindowName = "Input Video"
cv2.namedWindow(inputWindowName)

# Keys used for dict passed to the callbacks
backgroundImgKey = "backgroundImage"
inputFrameKey = "inputFrame"

# Min and max values for Green in HSV
minHue = 40
minSaturation = 100
minValue = 70

maxHue = 90
maxSaturation = 255
maxValue = 255

# Values needed for the Tolerance Trackbar
toleranceTrackbarName = "Tolerance"
maxToleranceValue = 25
toleranceValue = 5

# Values needed for the Softness Trackbar
softnessTrackbarName = "Softness"
maxSoftnessValue = 5
softnessValue = 0

# Details related to the Patches selected by the uder
topLeftPosition = []
bottomRightPosition = []


def calcMeanGreenPatchColour(inputHueChannel):
    """
    Takes the Hue channel of the input(HSV) image and calculates the mean value of green colour
    using the `topLeftPosition` and `bottomRightPosition` global variables.
    If the calculated mean colour isn't in the defined range for green colour, the default value is returned.

    :param inputHue: Hue channel of the HSV version of the frame being processed.
    """
    sum = 0.0
    pixelCount = 0

    if len(bottomRightPosition) == 0:
        return minHue + ((maxHue - minHue) / 2)

    for i in range(len(bottomRightPosition)):
        bottomRightPosition[i][0] += 1
        bottomRightPosition[i][1] += 1

        sum += np.sum(
            inputHueChannel[
                topLeftPosition[i][1] : bottomRightPosition[i][1],
                topLeftPosition[i][0] : bottomRightPosition[i][0],
            ]
        )
        pixelCount += (bottomRightPosition[i][0] - topLeftPosition[i][0]) * (
            bottomRightPosition[i][1] - topLeftPosition[i][1]
        )

    meanColourValue = int(sum / pixelCount)
    if meanColourValue < minHue or meanColourValue > maxHue:
        return minHue + ((maxHue - minHue) / 2)
    return meanColourValue


def calcMask(inputHSV, meanGreenColour):
    backgroundMask = cv2.inRange(
        inputHSV,
        (
            meanGreenColour - maxToleranceValue + toleranceValue,
            minSaturation,
            minValue,
        ),
        (
            meanGreenColour + maxToleranceValue - toleranceValue,
            maxSaturation,
            maxValue,
        ),
    )
    cv2.imshow("BG Mask", backgroundMask)
    if softnessValue > 0:
        kernelSize = (softnessValue * 2) + 1
        backgroundMask = cv2.GaussianBlur(backgroundMask, (kernelSize, kernelSize), 0)
        cv2.imshow("BG Mask - Blurred", backgroundMask)
    foregroundMask = cv2.bitwise_not(backgroundMask)
    cv2.imshow("FG Mask", foregroundMask)
    return backgroundMask, foregroundMask


def updateBackground(backgroundImg, inputFrame):
    """
    updateBackground updates the green screen/background of the input and replaces it with the provided background.
    Uses `cv2.imshow` to diplay the updated input.

    :param backgroundImg: The background image that will replace the green screen/background of the input.
    :param inputFrame: The input frame on which the green screen will be replaced.
    """

    inputHSV = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2HSV)
    meanGreenColour = calcMeanGreenPatchColour(inputHSV[:, :, 0])
    backgroundMask, foregroundMask = calcMask(inputHSV, meanGreenColour)

    updatedForeground = cv2.bitwise_and(inputFrame, inputFrame, mask=foregroundMask)
    updatedBackground = cv2.bitwise_and(
        backgroundImg, backgroundImg, mask=backgroundMask
    )
    updatedImg = cv2.bitwise_xor(updatedBackground, updatedForeground)
    cv2.imshow(outputWindowName, updatedImg)
    return


def onToleranceChanged(x, data):
    """
    onToleranceChanged is the callback used to handle the Tolerance trackbar on the input image.

    :param data: Contains the backgroundImage and inputImage that is needed to process the image.
    """
    global toleranceValue

    toleranceValue = cv2.getTrackbarPos(toleranceTrackbarName, inputWindowName)
    updateBackground(data[backgroundImgKey], data[inputFrameKey])


def onSoftnessChanged(x, data):
    """
    onSoftnessChanged is the callback used to handle the Softness trackbar on the input image.

    :param data: Contains the backgroundImage and inputImage that is needed to process the image.
    """
    global softnessValue

    softnessValue = cv2.getTrackbarPos(softnessTrackbarName, inputWindowName)
    updateBackground(data[backgroundImgKey], data[inputFrameKey])


def onMouseClicked(action, x, y, flags, data):
    """
    onMouseClicked is the callback used to handle the mouse action on the imput image.
    """
    if action == cv2.EVENT_LBUTTONDOWN:
        topLeftPosition.append([x, y])
    elif action == cv2.EVENT_LBUTTONUP:
        bottomRightPosition.append([x, y])
        updateBackground(data[backgroundImgKey], data[inputFrameKey])


def readAndResizeBackgrounds(inputWidth, inputHeight):
    """
    readAndResizeBackgrounds reads all the images in the `background` directory and resizes them
    to the input dimensions to use as the background for the input.

    :param inputWidth: width of the input frame
    :param inputHeight: height of the input frame
    """
    backgroundImages = []
    for path in glob.glob(DATA_PATH + "background/*.jpg"):
        background = cv2.imread(path)
        backgroundImages.append(cv2.resize(background, (inputWidth, inputHeight)))
    return backgroundImages


# cap = cv2.VideoCapture(DATA_PATH + "greenScreenInput/greenscreen-demo.mp4")
cap = cv2.VideoCapture(DATA_PATH + "greenScreenInput/greenscreen-asteroid.mp4")
if not cap.isOpened():
    print("Unable to open input video")
    sys.exit()

inputWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
inputHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
inputFrameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

backgrounds = readAndResizeBackgrounds(inputWidth, inputHeight)
backgroundDuration = inputFrameCount // len(backgrounds)
# print("#background: {} duration: {}".format(len(backgrounds), backgroundDuration))

frameCount = 0
backgroundIndex = 0
callbackData = {backgroundImgKey: [], inputFrameKey: []}
# while True:
#     hasFrame, frame = cap.read()
#     if not hasFrame:
#         break

#     if frameCount % backgroundDuration == 0:
#         backgroundIndex = (backgroundIndex + 1) % len(backgrounds)

#     callbackData[backgroundImgKey] = backgrounds[backgroundIndex]
#     callbackData[inputFrameKey] = frame

#     cv2.createTrackbar(
#         toleranceTrackbarName,
#         inputWindowName,
#         toleranceValue,
#         maxToleranceValue,
#         partial(onToleranceChanged, data=callbackData),
#         # lambda x: onToleranceChanged(x, backgroundImg=backgrounds[backgroundIndex]),
#         # onToleranceChanged(backgrounds[backgroundIndex]),
#     )
#     cv2.createTrackbar(
#         softnessTrackbarName,
#         inputWindowName,
#         softnessValue,
#         maxSoftnessValue,
#         partial(onSoftnessChanged, data=callbackData),
#     )
#     cv2.setMouseCallback(inputWindowName, onMouseClicked, callbackData)
#     cv2.imshow(inputWindowName, frame)

#     updateBackground(backgrounds[backgroundIndex], frame)
#     frameCount += 1

#     if cv2.waitKey(1) == 27:
#         break

hasFrame, frame = cap.read()

callbackData[backgroundImgKey] = backgrounds[backgroundIndex]
callbackData[inputFrameKey] = frame

cv2.createTrackbar(
    toleranceTrackbarName,
    inputWindowName,
    toleranceValue,
    maxToleranceValue,
    partial(onToleranceChanged, data=callbackData),
    # lambda x: onToleranceChanged(x, backgroundImg=backgrounds[backgroundIndex]),
    # onToleranceChanged(backgrounds[backgroundIndex]),
)
cv2.createTrackbar(
    softnessTrackbarName,
    inputWindowName,
    softnessValue,
    maxSoftnessValue,
    partial(onSoftnessChanged, data=callbackData),
)
cv2.setMouseCallback(inputWindowName, onMouseClicked, callbackData)
cv2.imshow(inputWindowName, frame)

updateBackground(backgrounds[backgroundIndex], frame)
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
