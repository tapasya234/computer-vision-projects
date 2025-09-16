import cv2
import numpy as np
import data_path
import glob
import sys

DETECTION_MODEL_EAST = "east"
DETECTION_MODEL_DB18 = "db18"
DETECTION_MODEL_DB50 = "db50"

# Set up the Text Detectors
inputSize = (320, 320)


def initDetectionModels(modelNames):
    if len(modelNames) == 0:
        print("No Models provided")
        sys.exit()

    binaryThreshold = 0.3
    polygonThreshold = 0.5
    meanDB = (122.67891434, 116.66876762, 104.00698793)

    detectorModels = list()
    for modelName in modelNames:
        if modelName == DETECTION_MODEL_EAST:
            textDetectorEast = cv2.dnn.TextDetectionModel_EAST(
                data_path.EAST_MODEL_PATH
            )
            textDetectorEast.setConfidenceThreshold(0.8).setNMSThreshold(0.4)
            textDetectorEast.setInputParams(
                1.0, inputSize, (123.68, 116.78, 103.94), True
            )
            detectorModels.append(textDetectorEast)
        elif modelName == DETECTION_MODEL_DB18:
            textDetectorDb18 = cv2.dnn.TextDetectionModel_DB(data_path.DB_18_MODEL_PATH)
            textDetectorDb18.setBinaryThreshold(binaryThreshold).setPolygonThreshold(
                polygonThreshold
            )
            textDetectorDb18.setInputParams(1.0 / 255, inputSize, meanDB, True)
            detectorModels.append(textDetectorDb18)
        else:
            textDetectorDb50 = cv2.dnn.TextDetectionModel_DB(data_path.DB_50_MODEL_PATH)
            textDetectorDb50.setBinaryThreshold(binaryThreshold).setPolygonThreshold(
                polygonThreshold
            )
            textDetectorDb50.setInputParams(1.0 / 255, inputSize, meanDB, True)
            detectorModels.append(textDetectorDb50)
    return detectorModels


def addBanner(img, bannerHeightPercentage=0.08):
    bannerHeight = int(img.shape[0] * bannerHeightPercentage)
    updatedImg = np.zeros(
        (bannerHeight + img.shape[0], img.shape[1], 3), dtype=np.uint8
    )
    updatedImg[bannerHeight:, :, :] = img
    return updatedImg


def addBannerText(img, label, locationXPercentage=0.04, locationYPercentage=0.35):
    locationX = int(locationXPercentage * img.shape[1])
    locationY = int(locationYPercentage * img.shape[0])
    cv2.putText(
        img, label, (locationX, locationY), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1
    )


def detectText(image, detectorModels: list[cv2.dnn.TextDetectionModel]):
    if len(detectorModels) == 0:
        print("No Models provided")
        sys.exit()
    # updated = addBanner(image)
    # fontThickness = int(0.01 * updated.shape[0])
    # print(fontThickness)

    images = list()
    colours = [(255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for i in range(len(detectorModels)):
        copy = image.copy()
        boxes, confidence = detectorModels[i].detect(copy)
        cv2.polylines(copy, boxes, True, colours[i % 3])
        images.append(copy)

    print(len(images))
    return images


detectorModels = initDetectionModels(
    [DETECTION_MODEL_DB50, DETECTION_MODEL_EAST, DETECTION_MODEL_DB18]
)

# img = cv2.imread(data_path.DATA_PATH + "input/image10.jpg")
# updated = detectText(img, detectorModels)
# cv2.imshow("Detected", updated)
# cv2.waitKey(0)

for path in glob.glob(data_path.DATA_PATH + "input/*.jpg"):
    img = cv2.imread(path)
    detectedImgList = detectText(img, detectorModels)
    cv2.imshow("Detected Text", np.hstack(detectedImgList))
    cv2.waitKey(0)

cv2.destroyAllWindows()
