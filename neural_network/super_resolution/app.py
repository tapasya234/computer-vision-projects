import cv2
import numpy as np
import glob
import data_path

superResolution = cv2.dnn_superres.DnnSuperResImpl().create()


def upscaleImage(img, method, scale):
    modelPath = data_path.DATA_PATH + "models/{}_x{}.pb".format(method, scale)
    superResolution.readModel(modelPath)

    superResolution.setModel(method.lower(), scale)

    return superResolution.upsample(img)


for path in glob.glob(data_path.DATA_PATH + "input/*.jpeg"):
    img = cv2.imread(path)
    cv2.imshow("Original", img)
    cv2.imshow("EDSR2", upscaleImage(img, "EDSR", 4))
    cv2.imshow("ESPCN2", upscaleImage(img, "ESPCN", 4))
    cv2.imshow("FRSCNN2", upscaleImage(img, "FSRCNN", 4))
    cv2.imshow("LapSRN2", upscaleImage(img, "LapSRN", 8))
    cv2.waitKey(0)

cv2.destroyAllWindows()
