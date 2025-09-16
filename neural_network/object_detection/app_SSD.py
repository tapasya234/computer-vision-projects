import cv2
import numpy as np
import data_path
import requests
from os import path
import glob
import display_test

if not path.exists(data_path.SSD_MODEL_PATH):
    print("Downloading MobileNet SSD Model.......")

    url = "https://opencv-courses.s3.us-west-2.amazonaws.com/ssd_mobilenet_frozen_inference_graph.pb"

    r = requests.get(url)

    with open(data_path.SSD_MODEL_PATH, "wb") as f:
        f.write(r.content)

    print("ssd_mobilenet_frozen_inference_graph Download complete!")

net = cv2.dnn.readNetFromTensorflow(
    model=data_path.SSD_MODEL_PATH, config=data_path.SSD_CONFIG_PATH
)

with open(data_path.SSD_CLASS_PATH) as fp:
    labels = fp.read().split("\n")
# print(sorted(labels))

dimensions = 300
mean = (0, 0, 0)


def detectObjects(net: cv2.dnn.Net, img: cv2.typing.MatLike):
    blob = cv2.dnn.blobFromImage(img, 1.0, (dimensions, dimensions), mean, True)
    net.setInput(blob)
    objects = net.forward()
    return objects


fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 2


def drawObjects(img, objects, threshold=0.25):
    height, width = img.shape[:2]

    for i in range(objects.shape[2]):
        classID = int(objects[0, 0, i, 1])
        score = objects[0, 0, i, 2]

        # Recover original cordinates from normalized coordinates
        x = int(objects[0, 0, i, 3] * width)
        y = int(objects[0, 0, i, 4] * height)
        w = int(objects[0, 0, i, 5] * width - x)
        h = int(objects[0, 0, i, 6] * height - y)

        if score > threshold:
            display_test.displayText(img, "{}".format(labels[classID]), x, y)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

    return img


# img = cv2.imread(data_path.DATA_PATH + "input/image2.jpg")
# detectedObjects = detectObjects(net, img)
# print(f"Detected {len(detectedObjects[0][0])} objects (no confidence filtering)")
# first_detected_obj = detectedObjects[0][0][0]
# print("First object:", first_detected_obj)
# detectedObjectsImg = drawObjects(img, detectedObjects, 0.4)
# cv2.imshow("Detected Objects", detectedObjectsImg)

images = []
detectedObjects = []
for path in glob.glob(data_path.DATA_PATH + "input/*.jpg"):
    img = cv2.imread(path)
    images.append(img)
    print("Classifying image ", path)
    detectedObjects.append(detectObjects(net, img))

# path = DATA_PATH + "input/image5.jpg"
# img = cv2.imread(path)
# images.append(img)
# print("Classifying image ", path)
# imagesClass.append(classify(img))

for i in range(len(images)):
    detectedImg = drawObjects(images[i], detectedObjects[i], 0.4)
    cv2.imshow("{}".format(i), detectedImg)
    cv2.waitKey(0)

# cv2.waitKey(0)
cv2.destroyAllWindows()
