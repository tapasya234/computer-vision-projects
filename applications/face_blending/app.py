import cv2
import numpy as np
from data_path import DATA_PATH
from matplotlib import pyplot as plt

# srcImg = cv2.imread(DATA_PATH + "input/female1_small.jpg")
# destinationImg = cv2.imread(DATA_PATH + "input/female3_small.jpg")
# # srcImg = cv2.imread(DATA_PATH + "input/female3_small.jpg")
# # destinationImg = cv2.imread(DATA_PATH + "input/female1_small.jpg")

# srcMask = np.zeros_like(srcImg, dtype=srcImg.dtype)
# points = np.array(
#     [
#         [239, 198],
#         [226, 232],
#         [213, 293],
#         [220, 361],
#         [281, 452],
#         [328, 459],
#         [401, 381],
#         [408, 287],
#         [392, 250],
#         [398, 229],
#         [357, 191],
#     ],
#     dtype=np.int32,
# )
# srcMask = cv2.fillPoly(srcMask, [points], (255, 255, 255))
# srcMask = cv2.morphologyEx(
#     srcMask,
#     cv2.MORPH_DILATE,
#     cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
# )
# cv2.imshow("Source Mask", srcMask)

# destinationMask = np.zeros_like(srcImg, dtype=destinationImg.dtype)
# points = np.array(
#     [
#         [237, 237],
#         [183, 327],
#         [178, 396],
#         [209, 504],
#         [248, 540],
#         [320, 540],
#         [368, 507],
#         [415, 394],
#         [409, 317],
#         [366, 252],
#     ],
#     dtype=np.int32,
# )
# destinationMask = cv2.fillPoly(destinationMask, [points], (255, 255, 255))
# cv2.imshow("Destination Mask", destinationMask)

# retval, maskThreshold = cv2.threshold(
#     destinationMask[:, :, 0], 128, 255, cv2.THRESH_BINARY
# )
# cv2.imshow("Threshold", maskThreshold)
# # print(srcMaskThreshold.shape)
# moments = cv2.moments(maskThreshold)
# center = (
#     int(moments["m01"] / moments["m00"]),
#     int(moments["m10"] / moments["m00"]),
# )
# print(center)

# cv2.imshow(
#     "Normal Cloning",
#     cv2.seamlessClone(
#         srcImg, destinationImg, destinationMask, center, cv2.NORMAL_CLONE
#     ),
# )
# # cv2.imshow(
# #     "Mixed Cloning",
# #     cv2.seamlessClone(srcImg, destinationImg, srcMask, center, cv2.MIXED_CLONE),
# # )
# plt.imshow(srcImg)
# plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows()

face = cv2.imread(DATA_PATH + "blemish.png")

grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
threshold = cv2.adaptiveThreshold(
    grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9
)

cloneArea = face[224:254, 269:299, :]
mask = np.ones_like(cloneArea) * 255

cv2.imshow("Blemish", face[224:254, 299:329, :])
cv2.imshow("Clone", cloneArea)
cv2.imshow("Mask", mask)
cv2.imshow(
    "Normal Cloning",
    cv2.seamlessClone(cloneArea, face, mask, (284, 239), cv2.NORMAL_CLONE),
)

cv2.waitKey(0)
cv2.destroyAllWindows()
