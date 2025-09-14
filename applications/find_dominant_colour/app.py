import cv2
from data_path import DATA_PATH
from matplotlib import pyplot
import numpy

img = cv2.imread(DATA_PATH + "jersey.jpg")
pyplot.figure(figsize=[20, 20])

# Convert to HSV
imageHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(imageHSV)
# print(H.shape)
# print(H)

H_array = H[S > 10].flatten()
# print(H_array.shape)
# print(H_array)


pyplot.subplot(121)
pyplot.imshow(img[:, :, ::-1])
pyplot.title("Jersey")
pyplot.axis("off")

pyplot.subplot(122)
pyplot.hist(H_array, bins=180, color="r")
pyplot.title("Histogram")

pyplot.show()
