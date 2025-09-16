from curses.textpad import rectangle
import cv2
import data_path
import numpy as np
import requests
from os import path
import display_test
import glob

# Initialize the parameters
objectnessThreshold = 0.5  # Objectness Threshold, high values filter out low objectness
confidenceThreshold = (
    0.5  # Confidence Threshold, high values filter out low confidence detections
)
# Non-Maximum suppression removes redundant overlapping bounding boxes.
nmsThreshold = (
    0.8  # Non-Maximum supression, higher values result in duplicate boxes per object
)
#  The below input values can be changed to 320 to get faster results or to 608 to get more accurate results.
inputWidth = 608  # Width of network's input image, larger is slower but more accurate
inputHeight = 608  # Height of network's input image, larger is slower but more accurate

# Load names of classes.
classes = None
with open(data_path.YOLO4_CLASS_PATH, "rt") as f:
    classes = f.read().rstrip("\n").split("\n")

# Give the configuration and weight files for the model and load the network using them.

if not path.exists(data_path.YOLO4_WEIGHTS_PATH):

    url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"

    r = requests.get(url)

    print("Downloading YOLO v4 Model.......")

    with open(data_path.YOLO4_WEIGHTS_PATH, "wb") as f:
        f.write(r.content)
    print("\nyolov4.weights Download complete!")


def getOutputNames(net: cv2.dnn.Net):
    layersName = net.getLayerNames()
    # print(layersName)

    return [layersName[i - 1] for i in net.getUnconnectedOutLayers()]


def displayDetectedObjects(img, outs):
    frame = img.copy()
    frameHeight, frameWidth = frame.shape[:2]

    classIDs = []
    confidenceList = []
    boxes = []

    for out in outs:
        for detection in out:
            if detection[4] > objectnessThreshold:
                scores = detection[5:]
                classID = np.argmax(scores)
                conf = scores[classID]
                if conf > confidenceThreshold:
                    centerX = int(detection[0] * frameWidth)
                    centerY = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)

                    left = int(centerX - width / 2)
                    top = int(centerY - height / 2)
                    classIDs.append(classID)
                    confidenceList.append(float(conf))
                    boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidenceList, confidenceThreshold, nmsThreshold)
    for i in indices:
        cv2.rectangle(
            frame,
            (boxes[i][0], boxes[i][1]),
            (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]),
            (255, 255, 255),
            2,
        )
        label = "{}:{:.2f}".format(classes[classIDs[i]], confidenceList[i])
        display_test.displayText(frame, label, boxes[i][0], boxes[i][1])
    return frame


net = cv2.dnn.readNetFromDarknet(
    data_path.YOLO4_CONFIG_PATH, data_path.YOLO4_WEIGHTS_PATH
)


def detectObjects(img, net: cv2.dnn.Net):
    blob = cv2.dnn.blobFromImage(
        img, 1 / 255, (inputWidth, inputHeight), (0, 0, 0), 1, crop=False
    )

    net.setInput(blob)
    return net.forward(getOutputNames(net))


# path = data_path.DATA_PATH + "input/image1.jpg"
# img = cv2.imread(path)
# print("Detecting objects:", path)
# outputs = detectObjects(img, net)
# detectedImg = displayDetectedObjects(img, outputs)
# cv2.imshow("Detected Img", detectedImg)
# cv2.waitKey(0)

images = []
detectedObjects = []
for path in glob.glob(data_path.DATA_PATH + "input/*.jpg"):
    img = cv2.imread(path)
    images.append(img)
    print("Detecting objects ", path)
    detectedObjects.append(detectObjects(img, net))

for i in range(len(images)):
    detectedImg = displayDetectedObjects(images[i], detectedObjects[i])
    cv2.imshow("{}".format(i), detectedImg)
    cv2.waitKey(0)


cv2.destroyAllWindows()
