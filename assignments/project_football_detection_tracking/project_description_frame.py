import cv2
import numpy as np

projectTitle = "DETECTION + TRACKING"
projectDescriptionPart1 = (
    "Given an input video, detect and track the soccer ball in the video."
)
projectDescriptionPart2 = "When the soccer ball is detected, the bounding box will in"
projectDescriptionPart3 = "and when it is tracked, the bounding box will in"
projectDescriptionPart4 = "When the ball is not detected or tracked,"
projectDescriptionPart5 = "text that states the same will be added to the banner in"
detectionDetails = "For detection, YOLOv4 TINY DNN is used."
trackingDetails = "For tracking, KCF method is used."
keymappingsInfo = "Press ENTER to start and press ESC to quit the program."

blueColourText = "BLUE colour"
greenColourText = "GREEN colour"
redColourText = "RED colour"

periodText = "."


def addTextToImg(
    img,
    text,
    orgPoint,
    fontFace=cv2.FONT_HERSHEY_PLAIN,
    fontColour=(255, 255, 255),
    fontScale=1.7,
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
    generateProjectDescriptionImage generated a image of the provided width and height
    and adds specific text to provide details about the project.

    :param inputWidth: Width of the project description image.
    :param inputHeight: Height of the project description image.
    """
    projectDescriptionImg = np.zeros((inputHeight, inputWidth, 3), dtype=np.uint8)
    addTextToImg(
        projectDescriptionImg,
        projectTitle,
        (int(inputWidth * 0.075), int(inputHeight * 0.15)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=3,
        fontColour=(0, 255, 255),
        fontThickness=2,
    )

    textOrgPointWidth = int(inputWidth * 0.10)
    textOrgPointHeightPercentage = 0.30

    addTextToImg(
        projectDescriptionImg,
        projectDescriptionPart1,
        (textOrgPointWidth, int(inputHeight * textOrgPointHeightPercentage)),
    )

    textOrgPointHeightPercentage += 0.07
    addTextToImg(
        projectDescriptionImg,
        projectDescriptionPart2,
        (textOrgPointWidth, int(inputHeight * textOrgPointHeightPercentage)),
    )
    addTextToImg(
        projectDescriptionImg,
        blueColourText,
        (int(inputWidth * 0.77), int(inputHeight * textOrgPointHeightPercentage)),
        fontColour=(185, 100, 0),
        fontThickness=2,
    )
    addTextToImg(
        projectDescriptionImg,
        periodText,
        (int(inputWidth * 0.9), int(inputHeight * textOrgPointHeightPercentage)),
    )

    textOrgPointHeightPercentage += 0.07
    addTextToImg(
        projectDescriptionImg,
        projectDescriptionPart3,
        (textOrgPointWidth, int(inputHeight * textOrgPointHeightPercentage)),
    )
    addTextToImg(
        projectDescriptionImg,
        greenColourText,
        (int(inputWidth * 0.65), int(inputHeight * textOrgPointHeightPercentage)),
        fontColour=(0, 255, 0),
        fontThickness=2,
    )
    addTextToImg(
        projectDescriptionImg,
        periodText,
        (int(inputWidth * 0.80), int(inputHeight * textOrgPointHeightPercentage)),
    )

    textOrgPointHeightPercentage += 0.07
    addTextToImg(
        projectDescriptionImg,
        projectDescriptionPart4,
        (textOrgPointWidth, int(inputHeight * textOrgPointHeightPercentage)),
    )

    textOrgPointHeightPercentage += 0.07
    addTextToImg(
        projectDescriptionImg,
        projectDescriptionPart5,
        (textOrgPointWidth, int(inputHeight * textOrgPointHeightPercentage)),
    )
    addTextToImg(
        projectDescriptionImg,
        redColourText,
        (int(inputWidth * 0.76), int(inputHeight * textOrgPointHeightPercentage)),
        fontColour=(0, 0, 255),
        fontThickness=2,
    )
    addTextToImg(
        projectDescriptionImg,
        periodText,
        (int(inputWidth * 0.88), int(inputHeight * textOrgPointHeightPercentage)),
    )

    textOrgPointHeightPercentage += 0.07
    for text in [
        detectionDetails,
        trackingDetails,
        keymappingsInfo,
    ]:
        addTextToImg(
            projectDescriptionImg,
            text,
            (textOrgPointWidth, int(inputHeight * textOrgPointHeightPercentage)),
        )
        textOrgPointHeightPercentage += 0.07

    cv2.imshow("Project Description", projectDescriptionImg)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
