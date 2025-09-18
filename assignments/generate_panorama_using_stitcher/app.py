import cv2
import numpy as np
import glob
from data_path import DATA_PATH
from matplotlib import pyplot as plt

imgFilePaths = [path for path in glob.glob(DATA_PATH + "scene/*jpg")]
imgFilePaths.sort()

images = []
for path in imgFilePaths:
    images.append(cv2.imread(path))


stitcher = cv2.Stitcher().create(cv2.STITCHER_PANORAMA)
# stitcher.setWaveCorrection(True)
# stitcher.setWa(cv2.detail.WAVE_CORRECT_VERT)
retval, pano = stitcher.stitch(images, pano=None)

result = pano[118:875, 20:2074]
# cv2.imshow("Pano 1", pano)
plt.imshow(result)
plt.show()
# retval, pano = stitcher.composePanorama(images, None)
# cv2.imshow("Pano 2", pano)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
