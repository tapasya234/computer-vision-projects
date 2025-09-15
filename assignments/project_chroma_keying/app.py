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

projectTitle = "Chroma-Keying"
projectDescriptionPart1 = "Given an input video with a green screen background, the program will automatically"
projectDescriptionPart2 = (
    "detect the green screen and replace it with the provided backgrounds."
)

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

# Values needed for the Background Trackbar
backgroundsDescription = (
    "Backgrounds: Decides the number of backgrounds to use. Minimun value is 1."
)
backgroundsTrackbarName = "Backgrounds"
maxBackgrounds = 5
backgroundsCount = 1

# Values needed for the Tolerance Trackbar
toleranceTrackbarDescription = (
    "Tolerance: Decides the range of green colour to use when remove the background."
)
toleranceTrackbarName = "Tolerance"
maxToleranceValue = 25
toleranceValue = 5

# Values needed for the Softness Trackbar
softnessTrackbarDescription = "Softness: Decides the value of softness (continuity) around the edges of the foreground."
softnessTrackbarName = "Softness"
maxSoftnessValue = 50
softnessValue = 0

# Values needed for the WaitTimer Trackbar
waitTimeTrackbarDescription = (
    "Wait Time: Decides the value of wait time for each frame."
)
waitTimerTrackbarName = "Wait Timer"
maxWaitTimerValue = 10
waitTimerValue = 1

# Details related to the Patches selected by the user
greenColourPatchesDescriptionPart1 = "Patches: The user can use the mouse to either select a single point or a specific patch"
greenColourPatchesDescriptionPart2 = (
    "on the input window to select mean green values to remove from the background."
)
topLeftPosition = []
bottomRightPosition = []

userControlsDescription = "The user can easily control various aspects of the project."


def addTextToImg(
    img,
    text,
    orgPoint,
    fontFace=cv2.FONT_HERSHEY_PLAIN,
    fontColour=(255, 255, 255),
    fontScale=1.3,
    fontThickness=1,
):
    """
    addTextToImg adds text to the provided image based on the parameters provided.

    :param img: Image on which the text will be added.
    :param text: The text to be added.
    :param orgPoint: Origin Point of the text. Expected to be a tuple of (x, y).
    :param fontFace: The style of font to use to style the text.
    :param fontColour: The colour of the font to use to style the text.
    :param fontScale: The scale of the font to use to style the text.
    :param fontThickness: The thickness of the font to use to style the text.
    """
    cv2.putText(
        img=img,
        text=text,
        org=orgPoint,
        fontFace=fontFace,
        fontScale=fontScale,
        color=fontColour,
        thickness=fontThickness,
        lineType=cv2.LINE_AA,
    )


def generateProjectDescriptionImage(inputWidth, inputHeight):
    """
    generateProjectDescriptionImage generated a image of the provided width and height and adds specific text to provide details about the project.

    :param inputWidth: Width of the project description image.
    :param inputHeight: Height of the project description image.
    """
    projectDescriptionImg = np.zeros((inputHeight, inputWidth, 3), dtype=np.uint8)
    addTextToImg(
        projectDescriptionImg,
        projectTitle,
        (int(inputWidth * 0.20), int(inputHeight * 0.10)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=3,
        fontColour=(0, 255, 0),
        fontThickness=2,
    )
    addTextToImg(
        projectDescriptionImg,
        projectDescriptionPart1,
        (int(inputWidth * 0.10), int(inputHeight * 0.20)),
    )
    addTextToImg(
        projectDescriptionImg,
        projectDescriptionPart2,
        (int(inputWidth * 0.10), int(inputHeight * 0.27)),
    )
    textOrgPointWidth = int(inputWidth * 0.10)
    textOrgPointHeightPercentage = 0.40
    for text in [
        userControlsDescription,
        greenColourPatchesDescriptionPart1,
        greenColourPatchesDescriptionPart2,
        backgroundsDescription,
        toleranceTrackbarDescription,
        softnessTrackbarDescription,
        waitTimeTrackbarDescription,
    ]:
        print(text)
        addTextToImg(
            projectDescriptionImg,
            text,
            (textOrgPointWidth, int(inputHeight * textOrgPointHeightPercentage)),
        )
        textOrgPointHeightPercentage += 0.07
        print(textOrgPointHeightPercentage)

    cv2.imshow("Project Description", projectDescriptionImg)
    cv2.waitKey(0)


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
    """
    calcMask calculated the foreground and background mask based on the provided green colour, toleranceValue and softnessValue.
    """
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

    foregroundMask = cv2.bitwise_not(backgroundMask)
    if softnessValue > 0:
        kernelSize = (softnessValue * 2) + 1
        foregroundMask = cv2.dilate(
            foregroundMask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize)),
        )
        foregroundMask = cv2.GaussianBlur(foregroundMask, (kernelSize, kernelSize), 0)
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
    return cv2.bitwise_xor(updatedBackground, updatedForeground)


def onBackgroundsChanged(*args):
    """
    onBackgroundsChanged is the callback used to handle the Backgrounds trackbar on the input image.

    :param data: Contains the backgroundImage and inputImage that is needed to process the image.
    """
    global backgroundsCount

    backgroundsCount = cv2.getTrackbarPos(backgroundsTrackbarName, inputWindowName)
    if backgroundsCount == 0:
        backgroundsCount = 1


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


def onWaitTimerChanged(*args):
    """
    onWaitTimerChanged is the callback used to handle the Wait Timer trackbar on the input image.
    It updates how long the Input and Output window should wait on every frame of the video. It doesn't affect the output video file generated.
    If value is set to 0, the window will wait indefinitely until a key press.
    """
    global waitTimerValue

    waitTimerValue = cv2.getTrackbarPos(waitTimerTrackbarName, inputWindowName)


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


cap = cv2.VideoCapture(DATA_PATH + "greenScreenInput/greenscreen-demo.mp4")
# cap = cv2.VideoCapture(DATA_PATH + "greenScreenInput/greenscreen-asteroid.mp4")
if not cap.isOpened():
    print("Unable to open input video")
    sys.exit()

inputWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
inputHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
inputFrameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

generateProjectDescriptionImage(inputWidth, inputHeight)

backgrounds = readAndResizeBackgrounds(inputWidth, inputHeight)
backgroundDuration = inputFrameCount // len(backgrounds)

videoWriter = cv2.VideoWriter(
    DATA_PATH + "chrome-keying-output.mp4",
    cv2.VideoWriter.fourcc(*"XVID"),
    # DATA_PATH + "chrome-keying-output.avi",
    # cv2.VideoWriter.fourcc("M", "J", "P", "G"),
    int(cap.get(cv2.CAP_PROP_FPS)),
    (inputWidth, inputHeight),
)

frameCount = 0
backgroundIndex = 0
callbackData = {backgroundImgKey: [], inputFrameKey: []}
while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        break

    if backgroundsCount > 1 and frameCount % backgroundDuration == 0:
        backgroundIndex = (backgroundIndex + 1) % backgroundsCount

    callbackData[backgroundImgKey] = backgrounds[backgroundIndex]
    callbackData[inputFrameKey] = frame

    cv2.createTrackbar(
        backgroundsTrackbarName,
        inputWindowName,
        backgroundsCount,
        maxBackgrounds,
        onBackgroundsChanged,
    )
    cv2.createTrackbar(
        toleranceTrackbarName,
        inputWindowName,
        toleranceValue,
        maxToleranceValue,
        partial(onToleranceChanged, data=callbackData),
    )
    cv2.createTrackbar(
        softnessTrackbarName,
        inputWindowName,
        softnessValue,
        maxSoftnessValue,
        partial(onSoftnessChanged, data=callbackData),
    )
    cv2.createTrackbar(
        waitTimerTrackbarName,
        inputWindowName,
        waitTimerValue,
        maxWaitTimerValue,
        onWaitTimerChanged,
    )
    cv2.setMouseCallback(inputWindowName, onMouseClicked, callbackData)
    cv2.imshow(inputWindowName, frame)

    updatedImg = updateBackground(backgrounds[backgroundIndex], frame)
    frameCount += 1

    cv2.imshow(outputWindowName, updatedImg)
    videoWriter.write(updatedImg)

    if cv2.waitKey(waitTimerValue * 10) == 27:
        break

cap.release()
videoWriter.release()
cv2.destroyAllWindows()
