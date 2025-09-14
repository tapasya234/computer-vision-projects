from operator import le
import cv2
from data_path import DATA_PATH
import numpy as np
from matplotlib import pyplot

# Implement Variance of absolute values of Laplacian - Method 1
# Input: image
# Output: Floating point number denoting the measure of sharpness of image

# Do NOT change the function name and definition


# Variance of absolute values of Laplacian:
# First apply Laplacian operator on the grayscaled image using Laplacian() of opencv then find it's absolute mean using NumPy mean function.
# Now apply the variance formula over the obtained absolute mean and absolute laplacian value. Return the sum of the variance.
def var_abs_laplacian(image):
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # imgBlurred = cv2.blur(imgGray, (3, 3))
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 1)

    imgLaplacian = cv2.Laplacian(imgBlurred, cv2.CV_32F) / 6
    mean = np.mean(imgLaplacian)
    # meanCV, stdDeviationCV = cv2.meanStdDev(imgLaplacian)
    laplacianVarian = np.divide(
        np.abs(imgLaplacian - mean), imgLaplacian.shape[0] * imgLaplacian.shape[1]
    )
    return np.sum(np.square(laplacianVarian))


def var_abs_gradient(image):
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # imgBlurred = cv2.blur(imgGray, (3, 3))
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgSobelX = cv2.Sobel(imgBlurred, cv2.CV_32F, 1, 0)
    imgSobelY = cv2.Sobel(imgBlurred, cv2.CV_32F, 0, 1)
    sobelVariance = np.sqrt(np.square(imgSobelX) + np.square(imgSobelY))
    return np.sum(np.square(sobelVariance))


# Implement Sum Modified Laplacian - Method 2
# Input: image
# Output: Floating point number denoting the measure of sharpness of image

# Do NOT change the function name and definition


# Convert the image to grayscale
# Take two kernels as arrays let it be x and y with 2D matrix as [[0, 0, 0], [-1, 2, -1], [0, 0, 0]] and [[0, -1, 0], [0, 2, 0], [0, -1, 0]]
# respectively or you may try some others kernel values as well.
# Now convolves X and Y image for both the kernels x and y separately using filter2D function of OpenCV.
# Now find the sum of abs(X) and abs(Y), let's say the result of its sum be S.
# At the end return the overall sum of array elements of S using sum()
def sum_modified_laplacian(im):
    imgGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.blur(imgGray, (3, 3))
    # imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 1)
    laplacianX = cv2.filter2D(
        imgBlurred,
        cv2.CV_32F,
        np.array(
            [
                [0, 0, 0],
                [-2, 4, -2],
                [0, 0, 0],
                # [0, -1, 0],
                # [0, 2, 0],
                # [0, -1, 0],
                # [0, 0, 0],
                # [-1, 2, -1],
                # [0, 0, 0],
                # [-1, -2, -1],
                # [0, 0, 0],
                # [1, 2, 1],
            ]
        ),
    )
    meanX = np.mean(laplacianX)
    laplacianY = cv2.filter2D(
        imgBlurred,
        cv2.CV_32F,
        np.array(
            [
                # [0, -1, 0],
                # [0, 2, 0],
                # [0, -1, 0],
                # [0, 0, 0],
                # [-1, 2, -1],
                # [0, 0, 0],
                # [1, 0, -1],
                # [2, 0, -2],
                # [1, 0, -1],
                [0, -2, 0],
                [0, 4, 0],
                [0, -2, 0],
            ]
        ),
    )
    meanY = np.mean(laplacianY)

    laplacian = np.sqrt(np.square(laplacianX - meanX) + np.square(laplacianY - meanY))
    return np.sqrt(np.sum(laplacian))
    # laplacian = np.abs(laplacianX) + np.abs(laplacianY)
    # return np.sum(laplacian)


# Read input video filename
filename = DATA_PATH + "focus-test.mp4"

# Create a VideoCapture object
cap = cv2.VideoCapture(filename)

# Read first frame from the video
ret, frame = cap.read()

# Display total number of frames in the video
print("Total number of frames : {}".format(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

maxV1 = 0
maxV2 = 0
maxV3 = 0

# Frame with maximum measure of focus
# Obtained using methods 1 and 2
bestFrame1 = 0
bestFrame2 = 0
bestFrame3 = 0

# Frame ID of frame with maximum measure
# of focus
# Obtained using methods 1 and 2
bestFrameId1 = 0
bestFrameId2 = 0
bestFrameId3 = 0

# Specify the ROI for flower in the frame
# UPDATE THE VALUES BELOW
top = 25
left = 455
bottom = 645
right = 1005

# pyplot.imshow(frame[top:bottom, left:right, -1])
# pyplot.show()

# # Iterate over all the frames present in the video
while ret:
    # Crop the flower region out of the frame
    flower = frame[top:bottom, left:right]
    # Get measures of focus from both methods
    val1 = var_abs_laplacian(flower)
    val2 = var_abs_gradient(flower)
    val3 = sum_modified_laplacian(flower)

    # get the current frame
    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    print("Frame: %d, VAR_LAP: %f, VAR_GRAD: %f SML: %d" % (frame_id, val1, val2, val3))

    # If the current measure of focus is greater
    # than the current maximum
    if val1 > maxV1:
        # Revise the current maximum
        maxV1 = val1
        # Get frame ID of the new best frame
        bestFrameId1 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # Revise the new best frame
        bestFrame1 = frame.copy()
        print("Frame ID of the best frame [Method 1]: {}".format(bestFrameId1))

    # If the current measure of focus is greater
    # than the current maximum
    if val2 > maxV2:
        # Revise the current maximum
        maxV2 = val2
        # Get frame ID of the new best frame
        bestFrameId2 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # Revise the new best frame
        bestFrame2 = frame.copy()
        print("Frame ID of the best frame [Method 2]: {}".format(bestFrameId2))

    # If the current measure of focus is greater
    # than the current maximum
    if val3 > maxV3:
        # Revise the current maximum
        maxV3 = val3
        # Get frame ID of the new best frame
        bestFrameId3 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # Revise the new best frame
        bestFrame3 = frame.copy()
        print("Frame ID of the best frame [Method 3]: {}".format(bestFrameId3))
    # Read a new frame
    ret, frame = cap.read()


print("================================================")
# Print the Frame ID of the best frame
print("Frame ID of the best frame [Method 1]: {}".format(bestFrameId1))
print("Frame ID of the best frame [Method 2]: {}".format(bestFrameId2))
print("Frame ID of the best frame [Method 3]: {}".format(bestFrameId3))

# Release the VideoCapture object
cap.release()

# Stack the best frames obtained using both methods
# out = np.hstack((bestFrame1, bestFrame2))

# Display the stacked frames
# plt.figure()
# plt.imshow(out[:, :, ::-1])
# plt.axis("off")
# cv2.imshow("BestFrames", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
