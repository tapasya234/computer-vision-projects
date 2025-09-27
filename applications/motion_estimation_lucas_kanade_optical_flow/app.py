import cv2
from data_path import DATA_PATH
import numpy as np

cap = cv2.VideoCapture(DATA_PATH + "cycling.mp4")
inputWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
inputHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# inputFPS = int(cap.get(cv2.CAP_PROP_FPS))

videoWriter = cv2.VideoWriter(
    DATA_PATH + "cycling_sparse-output.mp4",
    cv2.VideoWriter.fourcc(*"XVID"),
    20,
    (inputWidth, inputHeight),
)

# Parameters for Shi Tomasi (goodFeaturesToTrack) corner detection
numCorners = 100
feature_params = dict(
    # Max number of corners to be detected.
    maxCorners=numCorners,
    # Min quality of the image corners.
    # Calculated by multiplying the corner of highest value with this value and
    # the result is used as the minimum threshold.
    qualityLevel=0.3,
    # Min Euclidean distance between adjacent corners.
    minDistance=7,
    # Size of pixel neighborhood for computing a derivative covariation matrix.
    blockSize=7,
)

# Parameters for Lucas Kanade Optical Flow (calcOpticalFlowPyrLK)
opticalFlow_params = dict(
    # Size of window level at each pyramid level.
    winSize=(15, 15),
    # 0-based maximal pyramid level.
    maxLevel=2,
    # Specifies the termination criteria of the iterative search algorithm.
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.003),
)

# Create some random colours for the mostion estimation.
colours = np.random.randint(0, 255, (numCorners, 3))

# Take the first frame and find corners in it.
hasFrame, prevFrame = cap.read()
prevGray = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
prevPoints = cv2.goodFeaturesToTrack(prevGray, **feature_params)

# Create a mask for drawing the tracks
mask = np.zeros_like(prevFrame)
count = 0

while 1:
    hasFrame, frame = cap.read()
    if not hasFrame:
        break

    count += 1
    currentGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    currentPoints, status, err = cv2.calcOpticalFlowPyrLK(
        prevGray, currentGray, prevPoints, None, **opticalFlow_params
    )

    goodPrevPoints = prevPoints[status == 1]
    goodCurrentPoints = currentPoints[status == 1]

    for i, (current, previous) in enumerate(zip(goodCurrentPoints, goodPrevPoints)):
        a, b = current.ravel()
        c, d = previous.ravel()

        cv2.line(
            img=mask,
            pt1=(int(a), int(b)),
            pt2=(int(c), int(d)),
            color=colours[i].tolist(),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.circle(frame, (int(a), int(b)), 3, colours[i].tolist(), -1)

    # display every 5th frame
    displayFrame = cv2.add(frame, mask)
    videoWriter.write(displayFrame)

    cv2.imshow(str(count), displayFrame)

    # Update the previous frame and points
    prevGray = currentGray.copy()
    prevPoints = currentPoints.reshape(-1, 1, 2)

cap.release()
videoWriter.release()

cv2.waitKey(0)
cv2.destroyAllWindows()
