import cv2
import numpy
import os

inputFilePath = os.path.join(
    os.getcwd(),
    "python3/computer-vision-projects/applications/motion_detection/videos/motion_test.mp4",
)


def drawFrameBannerText(
    frame,
    bannerHeightPercentage=0.08,
    bannerText="",
    textColour=(0, 255, 0),
    fontScale=0.8,
    fontThickness=1,
):
    bannerHeight = int(frame.shape[0] * bannerHeightPercentage)
    cv2.rectangle(
        frame,
        pt1=(0, 0),
        pt2=(frame.shape[0], bannerHeight),
        color=(0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        frame,
        text=bannerText,
        org=(20, 10 + int((bannerHeightPercentage * frame.shape[0]) / 2)),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=fontScale,
        color=textColour,
        thickness=fontThickness,
        lineType=cv2.LINE_AA,
    )


cap = cv2.VideoCapture(inputFilePath)
if not cap.isOpened():
    print("Error: Unable to read/access file")
    exit()

frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameFPS = int(cap.get(cv2.CAP_PROP_FPS))

size = (frameWidth, frameHeight)
sizeQuad = (int(2 * frameWidth), int(2 * frameHeight))

videoWriter = cv2.VideoWriter(
    os.path.join(
        os.getcwd(),
        "python3/computer-vision-projects/applications/motion_detection/videos/motion_detected_output.mp4",
    ),
    cv2.VideoWriter.fourcc(*"XVID"),
    frameFPS,
    sizeQuad,
)

kernelSize = (5, 5)
redColour = (0, 0, 255)
yellowColour = (0, 255, 255)

bgSubtractor = cv2.createBackgroundSubtractorKNN(history=200)

updatedWinName = "Updated Preview"
cv2.namedWindow(updatedWinName)
while True:
    hasFrame, frame = cap.read()

    if not hasFrame or frame is None:
        break

    fgMask = bgSubtractor.apply(frame)
    cv2.imshow("FG Mask", fgMask)
    motionArea = cv2.findNonZero(fgMask)
    x, y, w, h = cv2.boundingRect(motionArea)

    if motionArea is not None:
        cv2.rectangle(
            frame,
            pt1=(x, y),
            pt2=(x + w, y + h),
            color=redColour,
            thickness=6,
        )
    fgMaskColour = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2RGB)
    drawFrameBannerText(fgMaskColour, bannerText="Foreground Mask")
    cv2.imshow("fgMaskColour", fgMaskColour)

    # Perform the same operations on an eroded foreground mask
    erodedFrame = frame.copy()
    erodedFgMask = cv2.erode(fgMask, numpy.ones(kernelSize, dtype=numpy.uint8))
    cv2.imshow("Eroded FG Mask", fgMask)
    erodedMotionArea = cv2.findNonZero(erodedFgMask)
    erodedX, erodedY, erodedW, erodedH = cv2.boundingRect(motionArea)

    if erodedMotionArea is not None:
        cv2.rectangle(
            erodedFrame,
            pt1=(erodedX, erodedY),
            pt2=(erodedX + erodedW, erodedY + erodedH),
            color=redColour,
            thickness=6,
        )
    erodedFgMaskColour = cv2.cvtColor(erodedFgMask, cv2.COLOR_GRAY2RGB)
    drawFrameBannerText(erodedFgMaskColour, bannerText="Eroded Foreground Mask")

    # Build the quad view
    frameTop = numpy.hstack([fgMaskColour, frame])
    compositeFrame = numpy.vstack(
        [
            [fgMaskColour, frame],
            [erodedFgMaskColour, erodedFrame],
        ]
    )

    cv2.line(
        compositeFrame,
        pt1=(0, int(compositeFrame.shape[0] / 2)),
        pt2=(compositeFrame.shape[1], int(compositeFrame.shape[0] / 2)),
        color=yellowColour,
        thickness=6,
        lineType=cv2.LINE_AA,
    )

    # videoWriter.write(compositeFrame)
    # cv2.imshow(updatedWinName, compositeFrame)
    # if cv2.waitKey(0) == 13 or cv2.waitKey(1) == 27:
    #     break

cap.release()
videoWriter.release()

cv2.destroyAllWindows()
