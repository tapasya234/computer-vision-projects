import cv2
import numpy as np


def computeHOG(hog: cv2.HOGDescriptor, images):
    hogFeatures = []
    for image in images:
        hogFeatures.append(hog.compute(image))

    return hogFeatures


def prepareSVMInputData(hogFeatures):
    """
    prepareSVMInputData modifies the hogFeatures into an acceptable input for SVM.

    :param hogFeatures: List for HOG Features for each input to SVM.
    """
    featureVectorsLen = len(hogFeatures)
    return np.float32(hogFeatures).reshape(-1, featureVectorsLen)


windowSize = (64, 128)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (4, 4)
nBins = 9
derivApeture = 1
windowSigma = -1
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = True
nLevels = 64
signedGradient = False

hog = cv2.HOGDescriptor(
    windowSigma,
    blockSize,
    blockStride,
    cellSize,
    nBins,
    derivApeture,
    windowSigma,
    histogramNormType,
    L2HysThreshold,
    gammaCorrection,
    nLevels,
    signedGradient,
)
