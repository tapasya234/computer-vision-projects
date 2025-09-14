import cv2
from data_path import DATA_PATH
import numpy

# 1. Create a threshold for lane lines
# 1. Selecting Region of Interest
# 1. Detecting Edges using Canny Edge Detector
# 1. Fit lines using Hough Line Transform
# 1. Extrapolate the lanes from lines found
# 1. Composite the result original frame


def drawLines(img, lines, color=[0, 0, 255], thickness=2):
    if lines is not None:
        for line in lines:
            print(line)
            x1, y1, x2, y2 = line
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def seperateLeftRightLines(lines):
    leftLines = []
    rightLines = []

    if lines is None:
        return leftLines, rightLines

    for line in lines:
        for x1, y1, x2, y2 in line:
            if y1 > y2:
                leftLines.append([x1, y1, x2, y2])
            elif y1 < y2:
                rightLines.append([x1, y1, x2, y2])

    return leftLines, rightLines


def calcAvg(values):
    n = len(values)
    if n == 0:
        n = 1
    return sum(values) / n


def extrapolateLines(lines, upperBorder, lowerBorder):
    slopes = []
    consts = []
    for line in lines:
        x1, y1, x2, y2 = line
        slope = (y2 - y1) / (x2 - x1)
        slopes.append(slope)
        consts.append((y1 - slope * x1))

    avgSlope = calcAvg(slopes)
    avgConst = calcAvg(consts)

    xLowerPoint = int((lowerBorder - avgConst) / avgSlope)
    xUpperPoint = int((upperBorder - avgConst) / avgSlope)

    return [xLowerPoint, lowerBorder, xUpperPoint, upperBorder]


def generateROI(img, vertices):
    mask = numpy.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(imgThresh, mask)


img = cv2.imread(DATA_PATH + "test_img.jpg")
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create the threshold of the image
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", imgGray)
_, imgThresh = cv2.threshold(imgGray, 150, 255, cv2.THRESH_BINARY)
# imgThresh = cv2.inRange(imgGray, 150, 255)
cv2.imshow("Thresh", imgThresh)

# Select the ROI
lanesROI = generateROI(
    imgThresh, numpy.array([[[100, 540], [900, 540], [515, 320], [450, 320]]])
)
cv2.imshow("ROI", lanesROI)

# Detect edges
imgEdges = cv2.Canny(lanesROI, 50, 100)
imgEdges = cv2.GaussianBlur(imgEdges, (7, 7), 3)
cv2.imshow("Smoothed Edges", imgEdges)

# Hough Transforms
lines = cv2.HoughLinesP(imgEdges, 1, numpy.pi / 180, 10, 5, 20)
lanesROI_detected = numpy.zeros((img.shape[0], img.shape[1], 3), numpy.uint8)
leftLines, rightLines = seperateLeftRightLines(lines)

# Extrapolate the lanes from lines
roiUpperBorder = 340
roiLowerBorder = 540

leftLane = extrapolateLines(leftLines, roiUpperBorder, roiLowerBorder)
rightLane = extrapolateLines(rightLines, roiUpperBorder, roiLowerBorder)

drawLines(lanesROI_detected, [leftLane], thickness=10)
drawLines(lanesROI_detected, [rightLane], thickness=10)
cv2.imshow("Detected Lanes", lanesROI_detected)

# Final image
annotatedImg = cv2.addWeighted(img, 0.8, lanesROI_detected, 1.0, 0)
cv2.imshow("Detected Lanes on actual image", annotatedImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
