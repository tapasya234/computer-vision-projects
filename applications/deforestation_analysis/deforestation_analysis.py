from typing import Sequence
import cv2
from matplotlib import pyplot
import os
import numpy


def readImage(imgName: str, imgType=cv2.IMREAD_COLOR) -> cv2.typing.MatLike:
    return cv2.imread(
        os.path.join(
            os.getcwd(),
            "python3/compVision/applications/deforestation_analysis/images/" + imgName,
        ),
        imgType,
    )


def plotRawImageBRGHisogram(img, title="", yscale="linear"):
    histB = cv2.calcHist([img], [0], None, [256], [0, 256])
    histG = cv2.calcHist([img], [1], None, [256], [0, 256])
    histR = cv2.calcHist([img], [2], None, [256], [0, 256])

    # Plot the histograms for each channel.
    fig = pyplot.figure(figsize=[15, 5])
    fig.suptitle(title)

    ax = fig.add_subplot(1, 3, 1)
    ax.set_yscale(yscale)
    pyplot.plot(histB, color="b", label="Blue")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(1, 3, 2)
    ax.set_yscale(yscale)
    pyplot.plot(histG, color="g", label="Green")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(1, 3, 3)
    ax.set_yscale(yscale)
    pyplot.plot(histR, color="r", label="Red")
    ax.grid()
    ax.legend()

    pyplot.show()


def detectGreenSegmentRatioHSV(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # plotRawImageBRGHisogram(imgHSV)
    lowerBound = numpy.array([36, 0, 50], dtype=numpy.uint8)
    upperBound = numpy.array([86, 250, 100], dtype=numpy.uint8)
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)

    landPixelsCount = cv2.countNonZero(mask)
    pixelsCount = mask.shape[0] * mask.shape[1]
    percent = round((landPixelsCount / pixelsCount) * 100, 2)
    return mask, percent


def detectGreenSegmentRatioBGR(img):
    # Create a mask for detecting the green segment using BGR
    lowerBoundG = numpy.array([0, 50, 0], dtype=numpy.uint8)
    upperBoundG = numpy.array([255, 100, 255], dtype=numpy.uint8)
    mask = cv2.inRange(img, lowerBoundG, upperBoundG)

    landPixelsCount = cv2.countNonZero(mask)
    pixelsCount = mask.shape[0] * mask.shape[1]
    percent = round((landPixelsCount / pixelsCount) * 100, 2)

    # fig = pyplot.figure(figsize=(20, 10))

    # ax = fig.add_subplot(1, 2, 1)
    # pyplot.imshow(img[:, :, ::-1])
    # ax.set_title("Original")

    # ax = fig.add_subplot(1, 2, 2)
    # pyplot.imshow(mask, cmap="gray")
    # ax.set_title("Color Segmented in Green Channel: " + str(percent) + "%")

    return mask, percent


def showAllFourImages(imgs, titles):
    # View all the images
    fig = pyplot.figure(figsize=[16, 8])

    for i, img in enumerate(imgs):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.set_title(titles[i])
        pyplot.imshow(img)


pyplot.rcParams["image.cmap"] = "gray"
img1 = readImage("1985.png")
img2 = readImage("1993.png")
img3 = readImage("2001.png")
img4 = readImage("2011.png")

showAllFourImages(
    [img1[:, :, ::-1], img2[:, :, ::-1], img3[:, :, ::-1], img4[:, :, ::-1]],
    ["1985", "1993", "2001", "2011"],
)
# b1, g1, r1 = cv2.split(img1)
# b4, g4, r4 = cv2.split(img4)

# plotRawImageBRGHisogram(img1, "1985", "log")
# plotRawImageBRGHisogram(img4, "2011", "log")
mask1, title1 = detectGreenSegmentRatioBGR(img1)
mask2, title2 = detectGreenSegmentRatioBGR(img2)
mask3, title3 = detectGreenSegmentRatioBGR(img3)
mask4, title4 = detectGreenSegmentRatioBGR(img4)
showAllFourImages([mask1, mask2, mask3, mask4], [title1, title2, title3, title4])

mask1, title1 = detectGreenSegmentRatioHSV(img1)
mask2, title2 = detectGreenSegmentRatioHSV(img2)
mask3, title3 = detectGreenSegmentRatioHSV(img3)
mask4, title4 = detectGreenSegmentRatioHSV(img4)
showAllFourImages([mask1, mask2, mask3, mask4], [title1, title2, title3, title4])

pyplot.show()
