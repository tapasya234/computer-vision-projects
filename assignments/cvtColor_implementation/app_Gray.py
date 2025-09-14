import cv2
import numpy
from data_path import DATA_PATH


def convertBGRtoGray(image):
    B, G, R = cv2.split(image)
    newImg = numpy.rint((0.299 * R) + (0.587 * G) + (0.114 * B))
    return numpy.uint8(numpy.clip(newImg, 0, 255))


img = cv2.imread(DATA_PATH + "sample.jpg")
imgGray = convertBGRtoGray(img)
imgGray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("User Defined", imgGray)
cv2.imshow("OpenCV", imgGray_cv)
cv2.imshow("Diff", numpy.abs(imgGray - imgGray_cv))

cv2.waitKey(0)
cv2.destroyAllWindows()
