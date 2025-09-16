import cv2
import numpy as np


def detectFaces(img: cv2.typing.MatLike, net: cv2.dnn.Net, detectionThreshold=0.7):
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1,
        size=(300, 300),
        mean=(104, 117, 123),
        swapRB=False,
        crop=False,
    )

    net.setInput(blob=blob)
    detections = net.forward()

    faces = []
    confidenceList = []

    imgHeight, imgWidth = img.shape[:2]
    for detection in detections[0][0]:
        confidence = detection[2]
        if confidence > detectionThreshold:
            confidenceList.append(confidence)

            left = detection[3] * imgWidth
            top = detection[4] * imgHeight
            right = detection[5] * imgWidth
            bottom = detection[6] * imgHeight

            faces.append((left, top, right - left, bottom - top))

    return np.array(faces).astype(int), np.array(confidenceList)
