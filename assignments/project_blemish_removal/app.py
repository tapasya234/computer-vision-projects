import cv2
from data_path import DATA_PATH
import numpy as np

# In OpenCV, you can remove blemishes from faces by
#  - Filtering(Apply image filtering techniques like Gaussian blur or median blur to smooth out blemishes),
#  - Thresholding(Use thresholding to segment blemishes from the background and then replace them with nearby skin tones),
#  - Morphological Operations(Utilise morphological operations like erosion and dilation to remove small blemishes or smooth out uneven areas),
#  - Contour Detection(Detect contours of blemishes and then either fill or replace them with nearby skin color).

# Constants to define the patch around the mouse click and also the
# neighbours-8 that are used to select the best patch around the blemish.
radius = 15
neightbours8 = np.array(
    [
        [-radius * 2, -radius * 2],
        [0, -radius * 2],
        [radius * 2, -radius * 2],
        [-radius * 2, 0],
        [radius * 2, 0],
        [-radius * 2, radius * 2],
        [0, radius * 2],
        [radius * 2, radius * 2],
    ]
)

# Reads the input and initializes variables related to it.
img = cv2.imread(DATA_PATH + "blemish.png")
# img = cv2.imread(DATA_PATH + "female2_small.jpg")
imgHeight, imgWidth = img.shape[:2]


def getClippedPoint(x, y):
    """
    getClippedPoint makes sure that the patch picked will be within the bounds of the image

    :param x: X value of the patch
    :param y: Y value of the patch
    """
    if x < radius:
        x = radius
    elif x > imgWidth - radius:
        x = imgWidth - radius - 1
    if y < radius:
        y = radius
    elif y > imgHeight - radius:
        y = imgHeight - radius - 1
    return x, y


def findBestPatchForCloning(blemishX, blemishY, imgThreshold):
    """
    findBestPatchForCloning finds the best patch on the threshold image using the neighbours-8 constant.

    :param blemishX: X position of the user-clicked blemish
    :param blemishY: Y position of the user-clicked blemish
    :param imgThreshold: threshold image of the input image
    """
    bestClonePoint = None
    leastNormalized = 1000
    copy = img.copy()
    for i in range(8):
        neighbourX, neighbourY = getClippedPoint(
            neightbours8[i][0] + blemishX,
            neightbours8[i][1] + blemishY,
        )

        area = imgThreshold[
            neighbourY - radius : neighbourY + radius,
            neighbourX - radius : neighbourX + radius,
        ]
        areaNormalized = np.sum(np.divide(area, 255.0))

        if areaNormalized < leastNormalized:
            leastNormalized = areaNormalized
            bestClonePoint = neighbourX, neighbourY

    cv2.circle(copy, bestClonePoint, radius, (0, 0, 0), 3)
    return bestClonePoint


# def checkIfAreaHasBlemish(x, y, frame):
#     """
#     checkIfAreaHasBlemish checks if the area picked has a blemish

#     :param x: Description
#     :param y: Description
#     :param frame: Description
#     """
#     area = frame[y - radius : y + radius, x - radius : x + radius]
#     print("(X, Y): {}".format(area[radius, radius]))
#     areaBlur = cv2.GaussianBlur(area, (3, 3), 0)

#     areaNormalized = area / 255.0
#     print(
#         "Area: {} Blur: {} Normalized: {}".format(
#             np.sum(area), np.sum(areaBlur), np.sum(areaNormalized)
#         )
#     )

#     separatorCol = np.ones_like(area, dtype=np.uint8) * 100
#     separatorRow = np.hstack([separatorCol, separatorCol, separatorCol])

#     areaSobelX = cv2.Sobel(area, cv2.CV_64F, 1, 0)
#     areaSobelX2 = cv2.Sobel(areaBlur, cv2.CV_64F, 1, 0)
#     print("SobelX GS: {} Blur: {}".format(np.sum(areaSobelX), np.sum(areaSobelX2)))

#     areaSobelY = cv2.Sobel(area, cv2.CV_64F, 0, 1)
#     areaSobelY2 = cv2.Sobel(areaBlur, cv2.CV_64F, 0, 1)
#     print("SobelY GS: {} Blur: {}".format(np.sum(areaSobelY), np.sum(areaSobelY2)))

#     areaLaplacian = cv2.Laplacian(area, cv2.CV_64F)
#     areaLaplacian2 = cv2.Laplacian(areaBlur, cv2.CV_64F)
#     print(
#         "Laplacian GS: {} Blur: {}".format(
#             np.sum(areaLaplacian), np.sum(areaLaplacian2)
#         )
#     )

#     edges = np.vstack(
#         [
#             np.hstack([areaSobelX, separatorCol, areaSobelX2]),
#             separatorRow,
#             np.hstack([areaSobelY, separatorCol, areaSobelY2]),
#             separatorRow,
#             np.hstack([areaLaplacian, separatorCol, areaLaplacian2]),
#         ]
#     )
#     cv2.imshow("Edges", edges)

#     findBestPatchForCloning((x, y), frame)
#     # sobelVariance = np.sqrt(np.square(areaSobelX) + np.square(areaSobelY))
#     # print("Sobel Variance: ", np.sum(np.square(sobelVariance)))
#     print("----------")


def removeBlemish(blemishX, blemishY):
    """
    removeBlemish updates the patch around the user-clicked blemish spot with the best patch around the blemish.

    :param blemishX: X position of the blemish picked by the user
    :param blemishY: Y position of the blemish picked by the user
    """
    global img
    imgGrayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgThreshold = cv2.adaptiveThreshold(
        imgGrayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9
    )

    clippedBlemishX, clippedBlemishY = getClippedPoint(blemishX, blemishY)
    clonePointX, clonePointY = findBestPatchForCloning(
        clippedBlemishX, clippedBlemishY, imgThreshold
    )
    mask = np.ones((30, 30), dtype=np.uint8) * 255

    # mask = imgThreshold[
    #     clippedBlemishY - radius : clippedBlemishY + radius,
    #     clippedBlemishX - radius : clippedBlemishX + radius,
    # ]

    img = cv2.seamlessClone(
        img[
            clonePointY - radius : clonePointY + radius,
            clonePointX - radius : clonePointX + radius,
            :,
        ],
        img,
        mask,
        (clippedBlemishX, clippedBlemishY),
        flags=cv2.NORMAL_CLONE,
    )


def onMouseClick(action, x, y, flags, data):
    """
    onMouseClick is the callback used to handle the mouse action on the imput image

    :param action: The mouse action performed on the image
    :param x: X positon of the mouse action on the image
    :param y: Y positon of the mouse action on the image
    """
    if action == cv2.EVENT_LBUTTONDOWN:
        area = img.copy()
        cv2.circle(area, (int(x), int(y)), radius, (0, 255, 0), 3)
        removeBlemish(x, y)
        cv2.imshow(windowName, area)


windowName = "Blemish Removal"
cv2.namedWindow(windowName)
cv2.setMouseCallback(windowName, onMouseClick)
cv2.imshow(windowName, img)

cv2.waitKey(0)
cv2.destroyAllWindows()
