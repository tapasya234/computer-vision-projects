import cv2
from data_path import DATA_PATH
import numpy as np

# Digital Video Stabilization: It is a process of stabilising the output video
# digitally without requiring any special sensors for estimating camera motion.
# There are three main steps:
#  1: Motion Estimation — Derives the transformation parametes between two consecutive frames.
#  2: Motion Smoothing - Filters out the unwanted motion by averaging neighbouring frames.
#  3: Image Composition - The stabilized video is reconstructed

# Larger the radius, more stable the video, but less reactive to sudden panning
SMOOTHING_RADIUS = 50

cap = cv2.VideoCapture(DATA_PATH + "piano.mp4")

inputWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
inputHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
inputFPS = int(cap.get(cv2.CAP_PROP_FPS))
inputFrameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

videoWriter = cv2.VideoWriter(
    DATA_PATH + "piano_stabilised.avi",
    cv2.VideoWriter.fourcc("M", "J", "P", "G"),
    inputFPS,
    (inputWidth * 2, inputHeight),
)

# MOTION ESTIMATION
# The Euclidean motion model requires that we know the motion of only 2 points
# in the two frames. However, in practice, it is a good idea to find the motion
# of 50-100 points, and then use them to robustly estimate the motion model.
_, prevFrame = cap.read()
prevGray = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)

# Store the tranformation matrix for motion in between two consecutive frames
transformValues = np.zeros((inputFrameCount - 1, 3), dtype=np.float32)

for i in range(inputFrameCount - 2):
    prevPoints = cv2.goodFeaturesToTrack(
        prevGray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3
    )

    hasFrame, currentFrame = cap.read()
    if not hasFrame:
        break

    currentGray = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow and filter valid points
    currentPoints, status, err = cv2.calcOpticalFlowPyrLK(
        prevGray, currentGray, prevPoints, None
    )
    assert prevPoints.shape == currentPoints.shape
    validIdx = np.where(status == 1)[0]
    prevPoints = prevPoints[validIdx]
    currentPoints = currentPoints[validIdx]

    # Calculate the transformation matrix for the
    # difference between the prev and current frames
    transformMatrix = cv2.estimateAffinePartial2D(prevPoints, currentPoints)

    # Translation
    dX = transformMatrix[0][0, 2]
    dY = transformMatrix[0][1, 2]
    # Rotation
    dA = np.arctan2(transformMatrix[0][1, 0], transformMatrix[0][0, 0])

    transformValues[i] = [dX, dY, dA]
    prevGray = currentGray

# MOTION SMOOTHING
# The easiest way to smooth any curve is to use a moving average filter.
# As the name suggests, a moving average filter replaces the value of a
# function at the point by the average of its neighbors defined by a window.


def movingAverage(curve, radius):
    windowSize = 2 * radius + 1
    f = np.ones(windowSize) / windowSize

    curvePadding = np.pad(curve, (radius, radius), "edge")
    curveSmoothed = np.convolve(curvePadding, f, "same")
    return curveSmoothed[radius:-radius]


def smooth(trajectory):
    smoothedTrajectory = np.copy(trajectory)

    # Filter the x, y and angle curves
    for i in range(3):
        smoothedTrajectory[:, i] = movingAverage(trajectory[:, i], SMOOTHING_RADIUS)

    return smoothedTrajectory


# Compute trajectory using cumulative sum of transformation matrices and smooth it.
trajectory = np.cumsum(transformValues, axis=0)
trajectorySmoothed = smooth(trajectory)
diff = trajectorySmoothed - trajectory
transformMatricesSmooth = transformValues + diff

# IMAGE COMPOSITION


# To stabilise a video, a frame may shrink in size which will lead to black boundary artifacts.
# This issue is mitigated by scaling the video about its center by a small amount.
def fixBorder(frame):
    size = frame.shape[:2]
    T = cv2.getRotationMatrix2D((size[1] / 2, size[0] / 2), 0, 1.04)
    return cv2.warpAffine(frame, T, (size[1], size[0]))


# Reset stream to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for i in range(inputFrameCount - 2):
    hasFrame, frame = cap.read()
    if not hasFrame:
        break

    dX = transformMatricesSmooth[i, 0]
    dY = transformMatricesSmooth[i, 1]
    dA = transformMatricesSmooth[i, 2]

    # Reconstruct transformation matrix using new values
    transformMatrix = np.zeros((2, 3), dtype=np.float32)
    transformMatrix[0, 0] = np.cos(dA)
    transformMatrix[0, 1] = -np.sin(dA)
    transformMatrix[1, 0] = np.sin(dA)
    transformMatrix[1, 1] = np.cos(dA)
    transformMatrix[0, 2] = dX
    transformMatrix[1, 2] = dY

    frameStabilised = cv2.warpAffine(frame, transformMatrix, (inputWidth, inputHeight))
    frameStabilised = fixBorder(frameStabilised)

    frameOutput = cv2.hconcat([frame, frameStabilised])

    # Resize the frame if it is too big
    if frameOutput.shape[1] > 1920:
        frameOutput = cv2.resize(frameOutput, (inputWidth, inputHeight))

    videoWriter.write(frameOutput)

cap.release()
videoWriter.release()
