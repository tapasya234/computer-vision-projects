import cv2
import numpy
from data_path import DATA_PATH

imagePath = DATA_PATH + "CoinsA.png"
image = cv2.imread(imagePath)

imageCopy = image.copy()
cv2.imshow("Original", image)

# Step 2.1: Convert Image to Grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", imageGray)

# Step 2.2: Split Image into R,G,B Channels
imageB, imageG, imageR = cv2.split(image)
# cv2.imshow("Blue", imageB)
# cv2.imshow("Green", imageG)
# cv2.imshow("Red", imageR)

# Step 3.1: Perform Thresholding¶
# You will have to carry out this step with different threshold values to see which one suits you the most.
# Do not remove those intermediate images and make sure to document your findings.
# _, imageThresh = cv2.threshold(imageGray, 100, 255, cv2.THRESH_TOZERO)
# _, imageThresh = cv2.threshold(imageGray, 100, 255, cv2.THRESH_TOZERO_INV)
# _, imageThresh = cv2.threshold(imageGray, 175, 255, cv2.THRESH_TOZERO_INV)
# _, imageThresh = cv2.threshold(imageGray, 75, 255, cv2.THRESH_TOZERO_INV)
# _, imageThresh = cv2.threshold(imageGray, 55, 255, cv2.THRESH_TOZERO_INV)
# _, imageThresh = cv2.threshold(imageGray, 65, 255, cv2.THRESH_TOZERO_INV)
# _, imageThresh = cv2.threshold(imageGray, 65, 255, cv2.THRESH_BINARY_INV)
# _, imageThresh = cv2.threshold(imageGray, 55, 255, cv2.THRESH_BINARY_INV)
# _, imageThresh = cv2.threshold(imageGray, 45, 255, cv2.THRESH_BINARY_INV)
# _, imageThresh = cv2.threshold(imageGray, 50, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("Threshold - Gray", imageThresh)

# _, imageThresh = cv2.threshold(imageG, 50, 255, cv2.THRESH_BINARY_INV)
# _, imageThresh = cv2.threshold(imageB, 25, 255, cv2.THRESH_BINARY_INV)
_, imageThresh = cv2.threshold(imageG, 25, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("Threshold - Blue", imageThresh)

# Step 3.2: Perform morphological operations
# You will have to carry out this step with different kernel size, kernel shape and morphological operations
# to see which one (or more) suits you the most. Do not remove those intermediate images and make sure to document your findings.
# closedImage = cv2.morphologyEx(
#     imageThresh,
#     cv2.MORPH_CLOSE,
#     cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
# )
# closedImage = cv2.morphologyEx(
#     imageThresh,
#     cv2.MORPH_CLOSE,
#     cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7)),
# )
dilatedImage = cv2.morphologyEx(
    imageThresh,
    cv2.MORPH_DILATE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
)
# dilatedImage = cv2.morphologyEx(
#     imageThresh,
#     cv2.MORPH_DILATE,
#     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
# )
# cv2.imshow("Dilated Image", dilatedImage)
# dilatedImage = cv2.morphologyEx(
#     dilatedImage,
#     cv2.MORPH_DILATE,
#     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
# )
# dilatedImage = cv2.morphologyEx(
#     dilatedImage,
#     cv2.MORPH_DILATE,
#     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
# )
# dilatedImage = cv2.morphologyEx(
#     dilatedImage,
#     cv2.MORPH_DILATE,
#     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
#     3,
# )
dilatedImage = cv2.morphologyEx(
    dilatedImage,
    cv2.MORPH_DILATE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    # 3,
)
# cv2.imshow("Dilated Image Image", dilatedImage)

# erodedImage = cv2.morphologyEx(
#     dilatedImage,
#     cv2.MORPH_ERODE,
#     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
#     3,
# )
# erodedImage = cv2.morphologyEx(
#     dilatedImage,
#     cv2.MORPH_ERODE,
#     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
#     3,
# )
# erodedImage = cv2.morphologyEx(
#     dilatedImage,
#     cv2.MORPH_ERODE,
#     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
#     3,
# )
# erodedImage = cv2.morphologyEx(
#     dilatedImage,
#     cv2.MORPH_ERODE,
#     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
#     5,
# )
erodedImage1 = cv2.morphologyEx(
    dilatedImage,
    cv2.MORPH_ERODE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    6,
)

# cv2.imshow("Eroded Image Image", erodedImage)
erodedImage = cv2.morphologyEx(
    erodedImage1,
    cv2.MORPH_ERODE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
)
cv2.imshow("Eroded Image", erodedImage)


# Step 4.1: Create SimpleBlobDetector
# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0

params.minDistBetweenBlobs = 2

# Filter by Area.
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.8

# Create SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)

# Step 4.2: Detect Coins
# Hints
# Use detector.detect(image) to detect the blobs (coins). The output of the function is a list of keypoints where each keypoint is unique for each blob.

# Print the number of coins detected as well.
keypoints = detector.detect(erodedImage)
print(len(keypoints))

imageBlobs = imageCopy
for keypoint in keypoints:
    x, y = keypoint.pt
    x = int(round(x))
    y = int(round(y))
    cv2.circle(imageBlobs, (x, y), 5, (255, 0, 0), -1)
    radius = int(keypoint.size // 2)
    cv2.circle(imageBlobs, (x, y), radius, (0, 255, 0), 2)
cv2.imshow("Blobs", imageBlobs)


# Step 4.4: Perform Connected Component Analysis
# In the final step, perform Connected Component Analysis (CCA) on the binary image to find out the number of connected components.
# Do you think we can use CCA to calculate number of coins? Why/why not?
def displayConnectedComponents(im):
    imLabels = im
    # The following line finds the min and max pixel values
    # and their locations in an image.
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imLabels)
    # Normalize the image so the min value is 0 and max value is 255.
    imLabels = 255 * (imLabels - minVal) / (maxVal - minVal)
    # Convert image to 8-bits unsigned type
    imLabels = numpy.uint8(imLabels)
    # Apply a color map
    imColorMap = cv2.applyColorMap(imLabels, cv2.COLORMAP_JET)
    # Display colormapped labels
    # plt.imshow(imColorMap[:, :, ::-1])
    cv2.imshow("Connected Components", imColorMap)


numConnected, imgLabels = cv2.connectedComponents(cv2.bitwise_not(erodedImage))
print(numConnected)
displayConnectedComponents(imgLabels)

imageContours = image.copy()
contours, hierarchy = cv2.findContours(
    erodedImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
)
print(len(contours))

# cv2.drawContours(imageContours, contours, -1, (0, 0, 0), 3)
# cv2.imshow("Contours", imageContours)


# External contours only
extContours, hierarchy = cv2.findContours(
    erodedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
print(len(extContours))

cv2.drawContours(imageContours, extContours, -1, (0, 0, 0), 3)
cv2.imshow("Contours", imageContours)

# Print Area and perimeter of contours
areasList = []
for i in range(len(contours)):
    areasList.append(cv2.contourArea(contours[i]))
    print(
        "Contour #{} has area: {} and perimeter: {}".format(
            i + 1,
            areasList[i],
            cv2.arcLength(contours[i], closed=True),
        )
    )

maxArea = 0
maxAreaIndex = 0
for i in range(len(areasList)):
    if areasList[i] > maxArea:
        maxArea = areasList[i]
        maxAreaIndex = i

print(maxArea)

imageContours = image.copy()
cv2.drawContours(imageContours, contours[:maxAreaIndex], -1, (0, 0, 0), 3)
if maxAreaIndex != len(contours) - 1:
    cv2.drawContours(imageContours, contours[maxAreaIndex + 1 :], -1, (0, 0, 0), 3)
cv2.imshow("Contours", imageContours)

imageContours = image.copy()
for i in range(len(contours)):
    if i == maxAreaIndex:
        continue

    (x, y), radius = cv2.minEnclosingCircle(contours[i])
    # cv2.circle(imageContours, (int(x), int(y)), 3, (0, 0, 255), -1)
    cv2.circle(imageContours, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.circle(imageContours, (int(x), int(y)), int(radius), (255, 0, 0), 5)
cv2.imshow("Contours", imageContours)

cv2.waitKey(0)
cv2.destroyAllWindows()
