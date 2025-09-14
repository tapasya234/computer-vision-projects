import cv2
import numpy
import data_path


# Resize the image to (50,50) and make it a three channel frame
def makeFrameVideoReady(img) -> cv2.typing.MatLike:
    img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_NEAREST)
    return cv2.merge([img, img, img])


def findMaxValue(img: cv2.typing.MatLike) -> int:
    minValue = max(img[0])
    for row in img[1:]:
        minValue = max(max(row), minValue)
    return minValue


def findMinValue(img: cv2.typing.MatLike) -> int:
    maxValue = min(img[0])
    for row in img[1:]:
        maxValue = min(min(row), maxValue)
    return maxValue


def findNeighbouringWhitePixel(img: cv2.typing.MatLike) -> bool:
    for row in img:
        for pixel in row:
            if pixel == 255:
                return True
    return False


img = numpy.zeros((10, 10), dtype=numpy.uint8)

img[0, 1] = 255
img[-1, 0] = 255
img[-2, -1] = 255
img[2, 2] = 255
img[5:8, 5:8] = 255

cv2.imshow("Original", makeFrameVideoReady(img))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) * 255
kernelHeight, kernelWidth = kernel.shape[:2]
imgHeight, imgWidth = img.shape[:2]
borderHorizontal = kernelWidth // 2
borderVertical = kernelHeight // 2

# Add padding around the image so that all pixels of the image can be used when performing the operations.
imgWithPadding = numpy.zeros(
    (imgHeight + kernelHeight, imgWidth + kernelWidth),
    dtype=numpy.uint8,
)
imgWithPadding = cv2.copyMakeBorder(
    img,
    borderVertical,
    borderVertical,
    borderHorizontal,
    borderHorizontal,
    cv2.BORDER_CONSTANT,
    value=0,
)

cv2.imshow("Original with Padding", makeFrameVideoReady(imgWithPadding))
cv2.imshow("cv2 Dilated", makeFrameVideoReady(cv2.dilate(img, kernel)))
cv2.imshow("cv2 Eroded", makeFrameVideoReady(cv2.erode(img, kernel)))


def dilate_centerPixel():
    dilatedImg = imgWithPadding.copy()
    videoWriter = cv2.VideoWriter(
        data_path.DATA_PATH + "dilation_center_process.avi",
        cv2.VideoWriter.fourcc("M", "J", "P", "G"),
        10,
        (50, 50),
    )

    # Add the original image as well as the image with padding
    videoWriter.write(makeFrameVideoReady(img))
    videoWriter.write(makeFrameVideoReady(dilatedImg))

    for i in range(borderVertical, imgHeight + borderVertical):
        for j in range(borderHorizontal, imgWidth + borderHorizontal):
            # Find a white pixel in the image
            if img[i - borderVertical, j - borderHorizontal]:

                dilatedImg[
                    i - borderVertical : i + borderVertical + 1,
                    j - borderHorizontal : j + borderHorizontal + 1,
                ] = cv2.bitwise_or(
                    dilatedImg[
                        i - borderVertical : i + borderVertical + 1,
                        j - borderHorizontal : j + borderHorizontal + 1,
                    ],
                    kernel,
                )
                videoWriter.write(makeFrameVideoReady(dilatedImg))

    # Get rid of the extra padding around the image
    dilatedImg = dilatedImg[
        borderVertical : imgHeight + borderVertical,
        borderHorizontal : imgWidth + borderHorizontal,
    ]

    cv2.imshow("Dilated - Center Pixel", makeFrameVideoReady(dilatedImg))
    videoWriter.write(makeFrameVideoReady(dilatedImg))
    videoWriter.release()


def dilate_neighbours():
    dilatedImg = imgWithPadding.copy()
    videoWriter = cv2.VideoWriter(
        data_path.DATA_PATH + "dilation_neighbours_process.avi",
        cv2.VideoWriter.fourcc("M", "J", "P", "G"),
        10,
        (50, 50),
    )
    videoWriter.write(makeFrameVideoReady(img))
    videoWriter.write(makeFrameVideoReady(dilatedImg))

    for i in range(borderVertical, imgHeight + borderVertical):
        for j in range(borderHorizontal, imgWidth + borderHorizontal):
            neighbourhood = imgWithPadding[
                i - borderVertical : i + borderVertical + 1,
                j - borderHorizontal : j + borderHorizontal + 1,
            ]
            if findNeighbouringWhitePixel(cv2.bitwise_and(neighbourhood, kernel)):
                dilatedImg[i, j] = findMaxValue(neighbourhood)
                videoWriter.write(makeFrameVideoReady(dilatedImg))

    dilatedImg = dilatedImg[
        borderVertical : imgHeight + borderVertical,
        borderHorizontal : imgWidth + borderHorizontal,
    ]

    cv2.imshow("Dilated - Neighbours", makeFrameVideoReady(dilatedImg))
    videoWriter.write(makeFrameVideoReady(dilatedImg))
    videoWriter.release()


def erode_centerPixel():
    erodedImg = imgWithPadding.copy()
    videoWriter = cv2.VideoWriter(
        data_path.DATA_PATH + "eroded_center_process.avi",
        cv2.VideoWriter.fourcc("M", "P", "E", "G"),
        10,
        (50, 50),
    )

    videoWriter.write(makeFrameVideoReady(img))
    videoWriter.write(makeFrameVideoReady(erodedImg))

    for i in range(borderVertical, imgHeight + borderVertical):
        for j in range(borderHorizontal, imgWidth + borderHorizontal):
            if img[i - borderVertical, j - borderHorizontal]:
                erodedImg[
                    i - borderVertical : i + borderVertical + 1,
                    j - borderHorizontal : j + borderHorizontal + 1,
                ] = cv2.bitwise_and(
                    erodedImg[
                        i - borderVertical : i + borderVertical + 1,
                        j - borderHorizontal : j + borderHorizontal + 1,
                    ],
                    kernel,
                )
                videoWriter.write(makeFrameVideoReady(erodedImg))
    erodedImg = erodedImg[
        borderVertical : imgHeight + borderVertical,
        borderHorizontal : imgWidth + borderHorizontal,
    ]

    cv2.imshow("Eroded  - Center Pixel", makeFrameVideoReady(erodedImg))
    videoWriter.write(makeFrameVideoReady(erodedImg))
    videoWriter.release()


def erode_neighbours():
    erodedImg = imgWithPadding.copy()
    videoWriter = cv2.VideoWriter(
        data_path.DATA_PATH + "erosion_neighbours_process.avi",
        cv2.VideoWriter.fourcc("M", "J", "P", "G"),
        10,
        (50, 50),
    )
    videoWriter.write(makeFrameVideoReady(img))
    videoWriter.write(makeFrameVideoReady(erodedImg))

    for i in range(borderVertical, imgHeight + borderVertical):
        for j in range(borderHorizontal, imgWidth + borderHorizontal):
            neighbourhood = imgWithPadding[
                i - borderVertical : i + borderVertical + 1,
                j - borderHorizontal : j + borderHorizontal + 1,
            ]
            if findNeighbouringWhitePixel(cv2.bitwise_and(neighbourhood, kernel)):
                erodedImg[i, j] = findMinValue(neighbourhood)
                videoWriter.write(makeFrameVideoReady(erodedImg))

    erodedImg = erodedImg[
        borderVertical : imgHeight + borderVertical,
        borderHorizontal : imgWidth + borderHorizontal,
    ]

    cv2.imshow("Eroded - Neighbours", makeFrameVideoReady(erodedImg))
    videoWriter.write(makeFrameVideoReady(erodedImg))
    videoWriter.release()


dilate_centerPixel()
dilate_neighbours()
erode_centerPixel()
erode_neighbours()
cv2.waitKey(0)
cv2.destroyAllWindows()
