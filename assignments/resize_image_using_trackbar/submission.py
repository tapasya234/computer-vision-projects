import cv2
import data_path

# MaxScaleValue is the max scale value that can be used to resize an image
maxScaleValue = 100
scaleValue = 1
trackbarValueName = "Scale Value"

# MaxScaleType is the type of scaling to be performed.
# `1` - Scale Down. `0` - Scale Up
maxScaleType = 1
scaleType = 0
trackbarTypeName = "Type\n0: Scale Up\n1: Scale Down"

img = cv2.imread(data_path.DATA_PATH + "truth.png")
winName = "Resize Window"
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)


def onScaleValueChanged(*args):
    global scaleType, scaleValue

    # Get the `scaleValue` from the trackbar and adjust based on `scaleType`
    scaleValue = args[0] / 100.0
    if scaleType == 0:
        scaleValue += 1
    else:
        scaleValue = 1 - scaleValue
    if scaleValue == 0:
        scaleValue = 1
    scaledImg = cv2.resize(
        img, None, fx=scaleValue, fy=scaleValue, interpolation=cv2.INTER_CUBIC
    )
    cv2.imshow(winName, scaledImg)


def onScaleTypeChanged(*args):
    global scaleType, scaleValue

    scaleType = args[0]
    scaleValue = 1
    cv2.setTrackbarPos(trackbarValueName, winName, 1)
    scaledImg = cv2.resize(
        img, None, fx=scaleValue, fy=scaleValue, interpolation=cv2.INTER_CUBIC
    )
    cv2.imshow(winName, scaledImg)


cv2.createTrackbar(
    trackbarValueName,
    winName,
    scaleValue,
    maxScaleValue,
    onScaleValueChanged,
)
cv2.createTrackbar(
    trackbarTypeName,
    winName,
    scaleType,
    maxScaleType,
    onScaleTypeChanged,
)

cv2.imshow(winName, img)
cv2.waitKey(0)
