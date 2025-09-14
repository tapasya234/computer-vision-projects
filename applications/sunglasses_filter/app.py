import cv2
import numpy
import data_path
from matplotlib import pyplot

muskImg = cv2.imread(data_path.DATA_PATH + "/images/musk.jpg")


def applySunglassesWithArithmatic(faceImg: cv2.typing.MatLike, opacity=255):
    faceImg = numpy.float32(faceImg) / 255
    faceWithGlasses = faceImg.copy()

    sunglassesImg = cv2.imread(
        data_path.DATA_PATH + "/images/sunglass.png", cv2.IMREAD_UNCHANGED
    )
    sunglassesImg = numpy.float32(sunglassesImg) / 255
    sunglassesImg = cv2.resize(sunglassesImg, None, fx=0.5, fy=0.5)
    sunglassesBGR = sunglassesImg[:, :, 0:3]
    sunglassesMask = cv2.merge(
        (
            sunglassesImg[:, :, 3] * (opacity / 255),
            sunglassesImg[:, :, 3] * (opacity / 255),
            sunglassesImg[:, :, 3] * (opacity / 255),
        )
    )

    topLeftRow = 130
    topLeftCol = 130
    bottomRightRow = topLeftRow + sunglassesBGR.shape[0]
    bottomRightCol = topLeftCol + sunglassesBGR.shape[1]

    eyesROI = faceWithGlasses[topLeftRow:bottomRightRow, topLeftCol:bottomRightCol]

    # (1-sunglassesMask) performs bitwise_not op
    # Multiply adds the sunglasses mask to the ROI
    maskedEye = cv2.multiply(eyesROI, (1 - sunglassesMask))

    # Use the mask to create the masked sunglass region
    maskedGlass = cv2.multiply(sunglassesBGR, sunglassesMask)

    updatedROI = cv2.add(maskedEye, maskedGlass)
    faceWithGlasses[topLeftRow:bottomRightRow, topLeftCol:bottomRightCol] = updatedROI
    cv2.imshow("With Sunglasses (Opacity " + str(opacity) + ")", faceWithGlasses)


def applyMustacheWithBitwise(faceImg: cv2.typing.MatLike):
    mustacheImg = cv2.imread(
        data_path.DATA_PATH + "/images/mustache.png", cv2.IMREAD_UNCHANGED
    )

    # Remove the padding around the mustache and resize it
    mustacheImg = mustacheImg[80:290, ...]
    mustacheImg = cv2.resize(mustacheImg, None, fx=0.28, fy=0.30)
    mustacheBGR = mustacheImg[:, :, 0:3]
    mustacheMask = cv2.merge(
        (
            mustacheImg[:, :, 3],
            mustacheImg[:, :, 3],
            mustacheImg[:, :, 3],
        )
    )

    topLeftRow = 250
    topLeftCol = 200
    bottomRightRow = topLeftRow + mustacheMask.shape[0]
    bottomRightCol = topLeftCol + mustacheMask.shape[1]
    roi = faceImg[topLeftRow:bottomRightRow, topLeftCol:bottomRightCol]
    maskedLips = cv2.bitwise_and(roi, cv2.bitwise_not(mustacheMask))
    maskedStache = cv2.bitwise_and(mustacheBGR, mustacheMask)
    updatedROI = cv2.bitwise_or(maskedLips, maskedStache)
    faceImg[topLeftRow:bottomRightRow, topLeftCol:bottomRightCol] = updatedROI


# applySunglassesWithArithmatic(muskImg)
# applySunglassesWithArithmatic(muskImg, opacity=200)
applyMustacheWithBitwise(muskImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
