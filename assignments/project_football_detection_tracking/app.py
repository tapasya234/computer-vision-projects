import sys
import cv2
import cv2.legacy

import data_path
import banner_utils
from detect_soccer_ball import detectSoccerBall
from project_description_frame import generateProjectDescriptionImage

DETECTION_COLOUR = (255, 0, 0)
TRACKING_COLOUR = (0, 255, 0)


def drawRectangle(frame: cv2.typing.MatLike, boundaryBox: list[int], rectangleColour):
    """
    Docstring for drawRectangleAroundSoccerBall

    :param frame: The frame on which a rectangle needs to be drawn
    :type frame: cv2.typing.MatLike
    :param boundaryBox: The boundary using which the rectangle will be drawn. It needs to be in the format of (left, top, width, height).
    :type boundaryBox: list[int]
    :param rectangleColour: The colour of the rectangle
    """
    cv2.rectangle(
        frame,
        (int(boundaryBox[0]), int(boundaryBox[1])),
        (int(boundaryBox[0] + boundaryBox[2]), int(boundaryBox[1] + boundaryBox[3])),
        rectangleColour,
        3,
        cv2.LINE_AA,
    )


net = cv2.dnn.readNetFromDarknet(
    cfgFile=data_path.CONFIG_PATH,
    darknetModel=data_path.MODEL_PATH,
)

cap = cv2.VideoCapture(data_path.DATA_PATH + "soccer-ball.mp4")
if not cap.isOpened():
    print("Unable to read/open file")
    sys.exit()

#  Read first frame and find the height of the frame after adding the banner
hasFrame, frame = cap.read()
if not hasFrame:
    print("Video has no frames")
    sys.exit()

frame = banner_utils.addBanner(frame)
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = frame.shape[0]

generateProjectDescriptionImage(frameWidth, frameHeight)
tracker = None

inputFPS = int(cap.get(cv2.CAP_PROP_FPS))
videoWriter = cv2.VideoWriter(
    data_path.DATA_PATH + "output.mp4",
    cv2.VideoWriter.fourcc(*"XVID"),
    inputFPS,
    (frameWidth, frameHeight),
)

bannerText = None
while cap.isOpened():
    hasFrame, frame = cap.read()
    if not hasFrame:
        break

    frame = banner_utils.addBanner(frame)

    if tracker is not None:
        didTrack, trackedBoundary = tracker.update(frame)
        if not didTrack:
            bannerText = "TRACK"
            tracker = None
            SOCCER_BALL_MEAN = None
        else:
            drawRectangle(frame, trackedBoundary, TRACKING_COLOUR)
            banner_utils.addText(frame, f"Tracked boundary: {detectedBoundary}")

    if tracker is None:
        detectedBoundary = detectSoccerBall(frame, net, frameWidth, frameHeight)
        if detectedBoundary is None:
            if detectedBoundary is not None:
                bannerText = "DETECT or " + bannerText
            else:
                bannerText = "DETECT"
        else:
            banner_utils.addText(frame, f"Detected boundary: {detectedBoundary}")
            drawRectangle(frame, detectedBoundary, DETECTION_COLOUR)
            tracker = cv2.legacy.TrackerKCF().create()
            tracker.init(frame, detectedBoundary)

    if bannerText is not None:
        banner_utils.addText(
            frame,
            f"Unable to {bannerText} soccer ball!",
            location=(50, 50),
            fontColour=(0, 0, 255),
        )
        bannerText = None

    cv2.imshow("Input", frame)
    videoWriter.write(frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
videoWriter.release()
cv2.destroyAllWindows()
