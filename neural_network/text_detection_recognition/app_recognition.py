import cv2
import numpy as np
import glob
import data_path

TEXT_RECOGNISER_SIZE = (100, 32)
TEXT_RECOGNISER_TARGET_VERTICES = np.float32(
    [
        [0, TEXT_RECOGNISER_SIZE[1] - 1],
        [0, 0],
        [TEXT_RECOGNISER_SIZE[0] - 1, 0],
        [TEXT_RECOGNISER_SIZE[0] - 1, TEXT_RECOGNISER_SIZE[1] - 1],
    ]
)


def initDetectionModel():
    textDetector = cv2.dnn.TextDetectionModel_DB(data_path.DB_50_MODEL_PATH)
    textDetector.setBinaryThreshold(0.3).setPolygonThreshold(0.5)
    textDetector.setInputParams(
        1 / 255.0, (640, 640), (122.67891434, 116.66876762, 104.00698793), True
    )
    return textDetector


def detectText(img, textDetector: cv2.dnn.TextDetectionModel_DB):
    image = img.copy()
    boxes, confidence = textDetector.detect(image)
    cv2.polylines(image, boxes, True, (255, 255, 0), 4)
    return boxes, image


def initRecognitionModel():
    vocabulary = []
    with open(data_path.VOCAB_PATH) as f:
        for l in f:
            vocabulary.append(l.strip())
        f.close()
    # print(vocabulary)
    # print(len(vocabulary))

    textRecogniser = cv2.dnn.TextRecognitionModel(data_path.CRNN_NET_OBJ_PATH)
    textRecogniser.setDecodeType("CTC-greedy")
    textRecogniser.setVocabulary(vocabulary)
    textRecogniser.setInputParams(
        1 / 127.5, TEXT_RECOGNISER_SIZE, (127.5, 127.5, 127.5), True
    )
    return textRecogniser


def fourPointsTransform(img, vertices):
    vertices = np.asarray(vertices).astype(np.float32)
    rotationMatrix = cv2.getPerspectiveTransform(
        vertices, TEXT_RECOGNISER_TARGET_VERTICES
    )
    return cv2.warpPerspective(img, rotationMatrix, TEXT_RECOGNISER_SIZE)


def recognizeText(img, verticesList, textRecogniser: cv2.dnn.TextRecognitionModel):
    textData = []
    outputCanvas = np.full((img.shape[0] * 2, img.shape[1] * 2, 3), 255, np.uint8)

    for vertices in verticesList:
        # Apply transformation on the bounding box detected by the text detection algorithm
        croppedROI = fourPointsTransform(img, vertices)
        recognisedResult = textRecogniser.recognize(croppedROI)

        textData.append(recognisedResult)

        boxHeight = int(abs(vertices[0, 1] - vertices[1, 1]))
        fontScale = cv2.getFontScaleFromHeight(
            cv2.FONT_HERSHEY_SIMPLEX, boxHeight - 10, 1
        )
        placement = (int(vertices[0, 0]), int(vertices[0, 1]))
        cv2.putText(
            outputCanvas,
            recognisedResult,
            placement,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            (255, 0, 0),
            1,
            5,
        )

    print(textData)
    return outputCanvas


# img = cv2.imread(data_path.DATA_PATH + "input/image1.jpg")
# detectorModel = initDetectionModel()
# boxes, detectedImg = detectText(img, detectorModel)
# recogniserModel = initRecognitionModel()
# recognisedImg = recognizeText(img, boxes, recogniserModel)
# cv2.imshow("Detected Text", detectedImg)
# cv2.imshow("Recognised Text", recognisedImg)
# cv2.waitKey(0)

for path in glob.glob(data_path.DATA_PATH + "input/*jpg"):
    img = cv2.imread(path)
    detectorModel = initDetectionModel()
    boxes, detectedImg = detectText(img, detectorModel)
    recogniserModel = initRecognitionModel()
    recognisedImg = recognizeText(img, boxes, recogniserModel)
    cv2.imshow("Detected Text", detectedImg)
    cv2.imshow("Recognised Text", recognisedImg)
    # cv2.imshow("Detected&Recognised Text", np.hstack([detectedImg, recognisedImg]))
    cv2.waitKey(0)

cv2.destroyAllWindows()
