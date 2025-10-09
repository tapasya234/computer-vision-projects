import cv2
import glob
from matplotlib import pyplot as plt

import data_path

NET_INPUT_SIZE = (368, 368)
nPoints = 15
POSE_PAIRS = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7],
    [1, 14],
    [14, 8],
    [8, 9],
    [9, 10],
    [14, 11],
    [11, 12],
    [12, 13],
]

net = cv2.dnn.readNetFromCaffe(
    prototxt=data_path.CONFIG_PATH, caffeModel=data_path.MODEL_PATH
)


def detectPoseEstimation(img: cv2.typing.MatLike):
    blob = cv2.dnn.blobFromImage(
        img,
        1.0 / 255,
        NET_INPUT_SIZE,
        [0, 0, 0],
        swapRB=True,
        crop=False,
    )

    net.setInput(blob)
    outputs = net.forward()

    plotPosePointsOnImage(img, outputs)


def displayProbabilityMaps(outputs, imgWidth, imgHeight):
    # Display probability maps
    for i in range(nPoints):
        probMap = outputs[0, i, :, :]
        displayMap = cv2.resize(probMap, (imgWidth, imgHeight), cv2.INTER_LINEAR)
        plt.subplot(2, 8, i + 1)
        plt.axis("off")
        plt.imshow(displayMap, cmap="jet")

    plt.show()


def extractPoints(outputs, imgWidth, imgHeight):
    scaleX = imgWidth / outputs.shape[3]
    scaleY = imgHeight / outputs.shape[2]

    points = []

    threshold = 0.1

    for i in range(nPoints):
        probMap = outputs[0, i, :, :]

        minValue, prob, maxValue, point = cv2.minMaxLoc(probMap)

        x = scaleX * point[0]
        y = scaleY * point[1]

        if prob > threshold:
            points.append((int(x), int(y)))
        else:
            points.append(None)

    return points


def plotPosePointsOnImage(img: cv2.typing.MatLike, outputs):
    imgHeight, imgWidth = img.shape[:2]

    imgPoints = img.copy()
    imgSkeleton = img.copy()

    points = extractPoints(outputs, imgWidth, imgHeight)
    for i, p in enumerate(points):
        cv2.circle(imgPoints, p, 8, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            imgPoints,
            f"{i}",
            p,
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(
                imgSkeleton, points[partA], points[partB], (255, 255, 0), 2, cv2.LINE_AA
            )

            cv2.circle(
                imgSkeleton,
                points[partA],
                8,
                (0, 0, 255),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    cv2.imshow("Points", imgPoints)
    cv2.imshow("Skeleton", imgSkeleton)

    cv2.waitKey(0)


for path in glob.glob(data_path.DATA_PATH + "input/*.jpg"):
    img = cv2.imread(path)
    detectPoseEstimation(img)

cv2.destroyAllWindows()
