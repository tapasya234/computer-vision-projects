import sys
import cv2
import cv2.legacy

import data_path
import banner_utils
from detect_soccer_ball import detectSoccerBall
from project_description_frame import generateProjectDescriptionImage

DETECTION_COLOUR = (255, 0, 0)
TRACKING_COLOUR = (0, 255, 0)

SOCCER_BALL_MEAN = 0.0
MEAN_TOLERANCE_VALUE = 15.0


def calcSoccerBallMean(frame: cv2.typing.MatLike, boundary: list):
    """
    calcSoccerBallMean calculates the mean of the eroded patch of the object
    which is supposed to be soccer ball.

    :param frame: The frame on which the soccer ball is detected/tracked.
    :type frame: cv2.typing.MatLike
    :param boundary: The boundary of the detected/tracked soccer ball. It should
    be in the form of (left, top, width, height).
    :type boundary: list
    """
    left, top, width, height = boundary
    if left < 0:
        left = 0

    if top < 0:
        top = 0

    patch = frame[
        int(top) : int(top + height),
        int(left) : int(left + width),
        :,
    ]
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, patch = cv2.threshold(patch, 150, 255, cv2.THRESH_BINARY)
    patch = cv2.dilate(patch, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return patch.mean()


def setSoccerBallMean(mean):
    """
    setSoccerBallMean sets the mean value of the gobal variable `SOCCER_BALL_MEAN`

    :param mean: Mean value that should be assigned to the global variable `SOCCER_BALL_MEAN`
    """
    global SOCCER_BALL_MEAN
    SOCCER_BALL_MEAN = mean


def shouldReDetectSoccerBall(frame: cv2.typing.MatLike, boundary: list):
    """
    shouldReDetectSoccerBall figures out if the soccer ball should be re-dectected
    if the mean of the tracked soccer ball is outside of the mean of the detected soccer ball.

    :param frame: The frame on which the soccer ball is detected/tracked.
    :type frame: cv2.typing.MatLike
    :param boundary: The boundary of the detected/tracked soccer ball. It should
    be in the form of (left, top, width, height).
    :type boundary: list
    """
    mean = calcSoccerBallMean(frame, boundary)
    if (
        mean < SOCCER_BALL_MEAN - MEAN_TOLERANCE_VALUE
        or mean > SOCCER_BALL_MEAN + MEAN_TOLERANCE_VALUE
    ):
        return True

    return False


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

projectDescriptionImg = generateProjectDescriptionImage(frameWidth, frameHeight)
tracker = None

inputFPS = int(cap.get(cv2.CAP_PROP_FPS))
videoWriter = cv2.VideoWriter(
    data_path.DATA_PATH + "output.mp4",
    cv2.VideoWriter.fourcc(*"XVID"),
    inputFPS,
    (frameWidth, frameHeight),
)

cv2.imshow("Project Description", projectDescriptionImg)
videoWriter.write(projectDescriptionImg)
cv2.waitKey(0)
cv2.destroyAllWindows()


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
        else:
            if shouldReDetectSoccerBall(frame, trackedBoundary):
                tracker = None
            else:
                drawRectangle(frame, trackedBoundary, TRACKING_COLOUR)
                banner_utils.addText(
                    frame,
                    f"Tracked boundary: {detectedBoundary}",
                    fontColour=(0, 255, 0),
                )

    if tracker is None:
        detectedBoundary = detectSoccerBall(frame, net, frameWidth, frameHeight)
        if detectedBoundary is None:
            bannerText = "DETECT"
        else:
            mean = calcSoccerBallMean(frame, detectedBoundary)
            if mean < 50:
                bannerText = "DETECT"
            else:
                bannerText = None
                setSoccerBallMean(mean)
                banner_utils.addText(
                    frame,
                    f"Detected boundary: {detectedBoundary}",
                    fontColour=(255, 100, 0),
                )
                drawRectangle(frame, detectedBoundary, DETECTION_COLOUR)
                # tracker = cv2.legacy.TrackerKCF().create()
                tracker = cv2.legacy.TrackerMOSSE().create()
                tracker.init(frame, detectedBoundary)

    if bannerText is not None:
        banner_utils.addText(
            frame,
            f"Unable to {bannerText} soccer ball!",
            location=(50, 50),
            fontColour=(0, 0, 255),
        )
        bannerText = None

    cv2.imshow("Detection + Tracking", frame)
    videoWriter.write(frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
videoWriter.release()
cv2.destroyAllWindows()
