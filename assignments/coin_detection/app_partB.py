import cv2
from data_path import DATA_PATH
import numpy

# Image path
imagePath = DATA_PATH + "CoinsB.png"

# Read image
# Store it in variable image
image = cv2.imread(imagePath)
cv2.imshow("Original", image)

# Convert to grayscale
# Store in variable imageGray
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Grayscale", imageGray)

# Split cell into channels
# Variables are: imageB, imageG, imageR
imageB = image[:, :, 0]
imageG = image[:, :, 1]
imageR = image[:, :, 2]
# cv2.imshow("Blue", imageB)
# cv2.imshow("Green", imageG)
# cv2.imshow("Red", imageR)

# Perform Thresholding
# You will have to carry out this step with different threshold values to see which one suits you the most.
# Do not remove those intermediate images and make sure to document your findings.
# _, imageThresh = cv2.threshold(imageGray, 75, 255, cv2.THRESH_BINARY)
# _, imageThresh = cv2.threshold(imageGray, 175, 255, cv2.THRESH_BINARY)
# _, imageThresh = cv2.threshold(imageGray, 125, 255, cv2.THRESH_BINARY)
# _, imageThresh = cv2.threshold(imageGray, 150, 255, cv2.THRESH_BINARY)
# _, imageThresh = cv2.threshold(imageGray, 165, 255, cv2.THRESH_BINARY)
# _, imageThresh = cv2.threshold(imageGray, 155, 255, cv2.THRESH_BINARY)
# _, imageThresh = cv2.threshold(imageGray, 160, 255, cv2.THRESH_BINARY)
# _, imageThresh = cv2.threshold(imageGray, 162, 255, cv2.THRESH_BINARY)
# _, imageThresh = cv2.threshold(imageGray, 161, 255, cv2.THRESH_BINARY)
# _, imageThresh = cv2.threshold(imageB, 161, 255, cv2.THRESH_BINARY)
# _, imageThresh = cv2.threshold(imageG, 161, 255, cv2.THRESH_BINARY)
# _, imageThresh = cv2.threshold(imageG, 158, 255, cv2.THRESH_BINARY)
# _, imageThresh = cv2.threshold(imageG, 160, 255, cv2.THRESH_BINARY)

imageBlurred = cv2.blur(imageGray, (3, 3))
# _, imageThresh = cv2.threshold(imageBlurred, 155, 255, cv2.THRESH_BINARY)
# _, imageThresh = cv2.threshold(imageBlurred, 160, 255, cv2.THRESH_BINARY)
_, imageThresh = cv2.threshold(imageBlurred, 158, 255, cv2.THRESH_BINARY)
# _, imageThresh = cv2.threshold(imageBlurred, 162, 255, cv2.THRESH_BINARY)
# _, imageThresh = cv2.threshold(imageBlurred, 161, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold", imageThresh)

# Perform morphological operations
# You will have to carry out this step with different kernel size, kernel shape
# and morphological operations to see which one (or more) suits you the most.
# Do not remove those intermediate images and make sure to document your findings.

# This it to clear the white dots in the coins
# imageOpened = cv2.morphologyEx(
#     imageThresh,
#     cv2.MORPH_OPEN,
#     kernel=cv2.getStructuringElement(cv2.MORPH_DIAMOND, (5, 5)),
# )
# cv2.imshow("Opened", imageOpened)

# imageOpened = cv2.morphologyEx(
#     imageThresh,
#     cv2.MORPH_OPEN,
#     kernel=cv2.getStructuringElement(cv2.MORPH_DIAMOND, (5, 5)),
#     iterations=2,
# )
# cv2.imshow("Opened I=2", imageOpened)

# imageOpened = cv2.morphologyEx(
#     imageThresh,
#     cv2.MORPH_OPEN,
#     kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
# )
# cv2.imshow("Opened Cross", imageOpened)

imageOpened = cv2.morphologyEx(
    imageThresh,
    cv2.MORPH_OPEN,
    kernel=cv2.getStructuringElement(cv2.MORPH_DIAMOND, (7, 7)),
    # iterations=2,
)
cv2.imshow("Coins with lesser white dots", imageOpened)

# This it to remove the black blobs in the background
# imageClosed = cv2.morphologyEx(
#     imageOpened,
#     cv2.MORPH_CLOSE,
#     kernel=cv2.getStructuringElement(cv2.MORPH_DIAMOND, (5, 5)),
# )
# cv2.imshow("Closed Diamond 5x5", imageClosed)

# imageClosed = cv2.morphologyEx(
#     imageOpened,
#     cv2.MORPH_CLOSE,
#     kernel=cv2.getStructuringElement(cv2.MORPH_DIAMOND, (7, 7)),
# )
# cv2.imshow("Closed Diamond 7x7", imageClosed)

# imageClosed = cv2.morphologyEx(
#     imageOpened,
#     cv2.MORPH_CLOSE,
#     kernel=cv2.getStructuringElement(cv2.MORPH_DIAMOND, (5, 5)),
#     iterations=3,
# )
# cv2.imshow("Closed Diamond 5x5  I=3", imageClosed)

imageClosed = cv2.morphologyEx(
    imageOpened,
    cv2.MORPH_CLOSE,
    kernel=cv2.getStructuringElement(cv2.MORPH_DIAMOND, (7, 7)),
    iterations=3,
)
cv2.imshow("Background with lesser black blobs", imageClosed)

# This is to comletely clear the white dots and fill the coins
# imageOpened = cv2.morphologyEx(
#     imageClosed,
#     cv2.MORPH_OPEN,
#     kernel=cv2.getStructuringElement(cv2.MORPH_DIAMOND, (7, 7)),
#     # iterations=3,
# )
# cv2.imshow("Opened2 7x7", imageOpened)
# imageOpened = cv2.morphologyEx(
#     imageClosed,
#     cv2.MORPH_OPEN,
#     kernel=cv2.getStructuringElement(cv2.MORPH_DIAMOND, (7, 7)),
#     iterations=3,
# )
# cv2.imshow("Opened2 7x7 I=3", imageOpened)
# imageOpened = cv2.morphologyEx(
#     imageClosed,
#     cv2.MORPH_OPEN,
#     kernel=cv2.getStructuringElement(cv2.MORPH_DIAMOND, (7, 7)),
#     iterations=5,
# )
# cv2.imshow("Opened2 7x7 I=5", imageOpened)
# imageOpened = cv2.morphologyEx(
#     imageClosed,
#     cv2.MORPH_OPEN,
#     kernel=cv2.getStructuringElement(cv2.MORPH_DIAMOND, (9, 9)),
#     iterations=3,
# )
# cv2.imshow("Opened2 9x9 I=3", imageOpened)
# imageOpened = cv2.morphologyEx(
#     imageClosed,
#     cv2.MORPH_OPEN,
#     kernel=cv2.getStructuringElement(cv2.MORPH_DIAMOND, (9, 9)),
#     iterations=5,
# )
# cv2.imshow("Opened2 9x9 I=5", imageOpened)
# imageOpened = cv2.morphologyEx(
#     imageClosed,
#     cv2.MORPH_OPEN,
#     kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7)),
#     iterations=9,
# )
# cv2.imshow("Opened2 7x7 I=9", imageOpened)
imageOpened = cv2.morphologyEx(
    imageClosed,
    cv2.MORPH_OPEN,
    kernel=cv2.getStructuringElement(cv2.MORPH_DIAMOND, (9, 9)),
    iterations=7,
)
cv2.imshow("Coins without any white dots", imageOpened)


# This is to completely remove the black blobs in the background
# imageClosed = cv2.morphologyEx(
#     imageOpened,
#     cv2.MORPH_CLOSE,
#     kernel=cv2.getStructuringElement(cv2.MORPH_DIAMOND, (7, 7)),
#     iterations=3,
# )
# cv2.imshow("Closed2 Diamond 7x7 I=3", imageClosed)

# imageClosed = cv2.morphologyEx(
#     imageOpened,
#     cv2.MORPH_CLOSE,
#     kernel=cv2.getStructuringElement(cv2.MORPH_DIAMOND, (7, 7)),
#     iterations=7,
# )
# cv2.imshow("Closed2 Diamond 7x7 I=7", imageClosed)

imageClosed = cv2.morphologyEx(
    imageOpened,
    cv2.MORPH_CLOSE,
    kernel=cv2.getStructuringElement(cv2.MORPH_DIAMOND, (7, 7)),
    iterations=9,
)

# imageClosed = cv2.morphologyEx(
#     imageOpened,
#     cv2.MORPH_CLOSE,
#     kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9)),
#     iterations=5,
# )
cv2.imshow("Background without any black blobs", imageClosed)

# Close the white gaps of the coins to convert the shape from donuts to circles
# imageOpened = cv2.morphologyEx(
#     imageClosed,
#     cv2.MORPH_OPEN,
#     kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9)),
#     iterations=10,
# )
# imageOpened = cv2.morphologyEx(
#     imageClosed,
#     cv2.MORPH_OPEN,
#     kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9)),
#     iterations=15,
# )
# imageOpened = cv2.morphologyEx(
#     imageClosed,
#     cv2.MORPH_OPEN,
#     kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9)),
#     iterations=12,
# )
imageOpened = cv2.morphologyEx(
    imageClosed,
    cv2.MORPH_OPEN,
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    iterations=13,
)
cv2.imshow("Donuts to Circles 1", imageOpened)

# imageOpened = cv2.morphologyEx(
#     imageOpened,
#     cv2.MORPH_OPEN,
#     kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9)),
#     iterations=18,
# )
# cv2.imshow("Donuts to Circles 2", imageOpened)

imageEroded = cv2.morphologyEx(
    imageOpened,
    cv2.MORPH_ERODE,
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    iterations=10,
)
cv2.imshow("Donuts to Circles 2", imageEroded)

imageEroded = cv2.morphologyEx(
    imageEroded,
    cv2.MORPH_ERODE,
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    iterations=6,
)
cv2.imshow("Donuts to Circles 3", imageEroded)

mask = cv2.bitwise_not(imageEroded)
cv2.imshow("Mask", mask)

maskDilated = cv2.morphologyEx(
    mask,
    cv2.MORPH_DILATE,
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    iterations=1,
)
cv2.imshow("Mask Eroded", maskDilated)


# Detect the coins using Blob Detector
imagesBlobs = image.copy()
params = cv2.SimpleBlobDetector_Params()
params.blobColor = 255
params.minDistBetweenBlobs = 1
# Filter by Area.
params.filterByArea = False
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.8
# Create SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)
# Detect Coins
keypoints = detector.detect(maskDilated)
print(len(keypoints))

for keypoint in keypoints:
    x, y = keypoint.pt
    x = int(round(x))
    y = int(round(y))
    cv2.circle(imagesBlobs, (x, y), 15, (255, 0, 0), -1)
    radius = int(keypoint.size // 2)
    cv2.circle(imagesBlobs, (x, y), radius, (255, 0, 255), 5)
cv2.imshow("Blobs", image)


# Detect the coins using ConnectComponentAnalysis
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
    cv2.imshow("Connected Components", imColorMap)
    # plt.imshow(imColorMap[:, :, ::-1])


numConnected, imgLabels = cv2.connectedComponents(maskDilated)
print(numConnected)
displayConnectedComponents(imgLabels)

# Detect the coins using Contours
imageContours = image.copy()
contours, hierarchy = cv2.findContours(
    maskDilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
)
print(len(contours))

# coinContours = list()
# # Calculate area and perimeter of contours
# for i, contour in enumerate(contours):
#     if cv2.contourArea(contour) > 100000:
#         coinContours.append(contour)
#     # print(
#     #     "Contour #{} has area: {} and perimeter: {} ".format(
#     #         i + 1,
#     #         cv2.contourArea(contour),
#     #         cv2.arcLength(contour, closed=True),
#     #     )
#     # )

# cv2.drawContours(imageContours, coinContours, -1, (0, 0, 255), 3, cv2.LINE_AA)
# cv2.imshow("Contours", imageContours)

coinContours = sorted(contours, key=lambda contour: cv2.contourArea(contour))
# for i, contour in enumerate(coinContours):
#     #     if cv2.contourArea(contour) > 100000:
#     #         coinContours.append(contour)
#     print(
#         "Contour #{} has area: {} and perimeter: {} ".format(
#             i + 1,
#             cv2.contourArea(contour),
#             cv2.arcLength(contour, closed=True),
#         )
#     )


coinContours = coinContours[2:]
# cv2.drawContours(imageContours, coinContours, -1, (0, 0, 255), 3, cv2.LINE_AA)
# cv2.imshow("Contours", imageContours)

imageContours = image.copy()
for index, contour in enumerate(coinContours):
    # We will use the contour moments
    # to find the centroid
    moments = cv2.moments(contour)
    x = int(round(moments["m10"] / moments["m00"]))
    y = int(round(moments["m01"] / moments["m00"]))

    # Mark the center
    cv2.circle(imageContours, (x, y), 20, (255, 0, 0), -1)

    # Mark the contour number
    cv2.putText(
        imageContours,
        "{}".format(index + 1),
        (x + 40, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (0, 0, 255),
        thickness=10,
        lineType=cv2.LINE_AA,
    )
cv2.imshow("Size Ordered Contours", imageContours)

imageContours = image.copy()
for index, contour in enumerate(coinContours):
    # We will use the contour moments
    # to find the centroid
    # moments = cv2.moments(contour)
    # x = int(round(moments["m10"] / moments["m00"]))
    # y = int(round(moments["m01"] / moments["m00"]))

    (x, y), radius = cv2.minEnclosingCircle(contour)
    # Mark the center
    cv2.circle(
        imageContours, (int(x), int(y)), int(round(radius)) - 75, (255, 0, 0), 10
    )

    # Mark the contour number
    # cv2.putText(
    #     imageContours,
    #     "{}".format(index + 1),
    #     (x + 40, y - 10),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     3,
    #     (0, 0, 255),
    #     thickness=10,
    #     lineType=cv2.LINE_AA,
    # )
cv2.imshow("Size Ordered Contours", imageContours)

cv2.waitKey(0)
cv2.destroyAllWindows()
