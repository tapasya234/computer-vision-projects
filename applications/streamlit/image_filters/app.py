from turtle import width
import streamlit
import cv2
import numpy
from io import BytesIO
import base64
from PIL import Image

# Constants for filters
blackWhite = 0
blur = 1
sharpen = 2
sepia = 3
vignette = 4
edges = 5
embossedEdges = 6
exposure = 7
outlines = 8
pencilSketch = 9
stylized = 10


def applyFilter(img, filterType):
    if filterType == blackWhite:
        return "Black&White", grayscaleImage(img)
    if filterType == blur:
        return "Blur", blurImage(img)
    if filterType == sharpen:
        return "Sharpen", sharpenImage(img)
    if filterType == sepia:
        return "Sepia", sepiaImage(img)
    if filterType == vignette:
        return "Vignette", vignetteImage(img)
    if filterType == edges:
        return "Edges", edgesImage(img)
    if filterType == embossedEdges:
        return "Embossed Edges", embossedEdgesImg(img)
    if filterType == exposure:
        return "Exposure", grayscaleImage(img)
    if filterType == outlines:
        return "Outlines", outlineImg(img)
    if filterType == pencilSketch:
        return "Pencil", pencilSketchImg(img)
    if filterType == stylized:
        return "Stylized", stylizedImg(img)


def grayscaleImage(img: cv2.typing.MatLike):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Blur the image. Add customization option
def blurImage(img: cv2.typing.MatLike):
    return cv2.blur(img, (7, 7))


# Sharpen the image. Add customization option
def sharpenImage(img: cv2.typing.MatLike):
    kernalSharpening = numpy.array(
        [
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1],
        ]
    )
    return cv2.filter2D(img, -1, kernalSharpening)


def sepiaImage(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRGB = numpy.array(imgRGB, dtype=numpy.float64)
    imgTransformed = cv2.transform(
        imgRGB,
        numpy.matrix(
            [
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131],
            ]
        ),
    )
    imgTransformed = numpy.clip(imgTransformed, 0, 255)
    imgTransformed = numpy.array(imgTransformed, dtype=numpy.uint8)
    return cv2.cvtColor(imgTransformed, cv2.COLOR_RGB2BGR)


# Dims the image in a Gaussian graph format. Add customization option
def vignetteImage(img: cv2.typing.MatLike, level=2) -> cv2.typing.MatLike:
    height, width = img.shape[:2]

    kernelX = cv2.getGaussianKernel(width, width / level)
    kernelY = cv2.getGaussianKernel(height, height / level)

    kernel = kernelY * kernelX.T
    mask = kernel / kernel.max()

    outputImg = img.copy()
    for i in range(3):
        outputImg[:, :, i] = outputImg[:, :, i] * mask
    return outputImg


def edgesImage(img: cv2.typing.MatLike, shouldBlur=True) -> cv2.typing.MatLike:
    grayImg = grayscaleImage(img)
    blurImg = blurImage(grayImg) if shouldBlur else grayImg
    return cv2.Canny(blurImg, 140, 200)


def embossedEdgesImg(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    return cv2.filter2D(
        img,
        -1,
        numpy.array(
            [
                [0, -3, -3],
                [3, 0, -3],
                [3, 3, 0],
            ]
        ),
    )


def outlineImg(img: cv2.typing.MatLike, k=10) -> cv2.typing.MatLike:
    return cv2.filter2D(
        img,
        -1,
        numpy.array(
            [
                [-1, -1, -1],
                [-1, k, -1],
                [-1, -1, -1],
            ]
        ),
    )


def pencilSketchImg(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    blurImg = cv2.GaussianBlur(img, (5, 5), 0, 0)
    sketchGS, sketchColor = cv2.pencilSketch(img)
    return sketchColor


def stylizedImg(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    blurImg = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.stylization(blurImg, sigma_s=40, sigma_r=0.1)


streamlit.title("Image Filtering")
streamlit.text(
    "Allows you to apply different types of filters to image and downdald them."
)

rawImg = streamlit.file_uploader("Upload an image", ["JPEG", "JPG", "PNG"])

if rawImg is None:
    streamlit.text("Unable to read image")
else:
    # Read the file and convert it into an OpenCV Image
    rawBytes = numpy.asarray(bytearray(rawImg.read()), dtype=numpy.uint8)
    inputImg = cv2.imdecode(rawBytes, flags=cv2.IMREAD_COLOR)
    streamlit.image(inputImg, caption="Input Image")

    for filter in range(11):
        filterName, outputImg = applyFilter(inputImg, filter)
        streamlit.image(
            Image.fromarray(outputImg), caption="With " + filterName + " filter applied"
        )
