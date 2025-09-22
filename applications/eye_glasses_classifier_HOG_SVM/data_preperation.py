import cv2
import os
from data_path import DATA_PATH

pathGlasses = DATA_PATH + "glassesDataset/cropped_withGlasses2"
pathNoGlasses = DATA_PATH + "glassesDataset/cropped_withoutGlasses2"


def retrieveImageFilenamesFromDataset(datasetPath):
    inputDir = os.path.expanduser(datasetPath)
    if os.path.isdir(inputDir):
        imagesFilenames = [
            f for f in os.listdir(inputDir) if f.lower().endswith(".jpg")
        ]
        # imagesFilenames = os.listdir(inputDir)
        imagesFilenames.sort()
        # print(imagesFilenames)
    # print(len(imagesFilenames))
    return inputDir, imagesFilenames


# The below function reads the images in a specified directory.
# It creates two lists for storing train and test images.
# Also creates lists for train and test labels.
# The number of testing images is kept at 20% of the total number of images.
def getTrainTestData(datasetPath, classValue, testFraction=0.20):
    trainData = []
    trainLabels = []

    testData = []
    testLabels = []

    inputDir, fileNames = retrieveImageFilenamesFromDataset(datasetPath)

    # Get images from the dataset and find number of train and test samples
    testSampleCount = int(len(fileNames) * testFraction)

    for counter, imgFileName in enumerate(fileNames):
        img = cv2.imread(os.path.join(inputDir, imgFileName))
        if counter < testSampleCount:
            testData.append(img)
            testLabels.append(classValue)
        else:
            trainData.append(img)
            trainLabels.append(classValue)

    return trainData, trainLabels, testData, testLabels
