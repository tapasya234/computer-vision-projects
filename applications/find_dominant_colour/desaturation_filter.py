import cv2
from data_path import DATA_PATH
import numpy

img = cv2.imread(DATA_PATH + "jersey.jpg")
cv2.imshow("Original", img)

saturationScale = 300

imageHSV = numpy.float32(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
H, S, V = cv2.split(imageHSV)

newS = numpy.clip(S * saturationScale, 0, 255)
updatedImageHSV = numpy.uint8(cv2.merge([H, newS, V]))

cv2.imshow("Desaturation Filter", cv2.cvtColor(updatedImageHSV, cv2.COLOR_HSV2BGR))

cv2.waitKey(0)
cv2.destroyAllWindows()
