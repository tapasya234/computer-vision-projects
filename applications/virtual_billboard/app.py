import cv2
import numpy
from data_path import DATA_PATH

windowName = "Billboard"


def onMouseClicked(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        img = data["img"]
        cv2.circle(img, (x, y), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.imshow(windowName, img)

        if len(data["points"]) < 4:
            data["points"].append([x, y])


def getROIPointsOfBillboard(img):
    data = {"img": img, "points": []}
    cv2.imshow(windowName, img)
    cv2.setMouseCallback(windowName, onMouseClicked, data)

    cv2.waitKey(0)

    return numpy.vstack(data["points"]).astype(float)


imgBill = cv2.imread(DATA_PATH + "Apollo-8-Launch.png")
imgBillboard = cv2.imread(DATA_PATH + "times_square.jpg")

height, width = imgBill.shape[:2]
pointsBill = numpy.array(
    [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
    dtype=float,
)

pointsBillboard = getROIPointsOfBillboard(imgBillboard)

h, mask = cv2.findHomography(pointsBill, pointsBillboard)
warpedBill = cv2.warpPerspective(
    imgBill, h, (imgBillboard.shape[1], imgBillboard.shape[0])
)

cv2.fillConvexPoly(imgBillboard, pointsBillboard.astype(int), 0, 16)
imgBillboard = imgBillboard + warpedBill

cv2.imshow(windowName, imgBillboard)
cv2.waitKey(0)
cv2.destroyAllWindows()
