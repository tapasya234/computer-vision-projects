import cv2
import data_path
import numpy
import face_detection

net = cv2.dnn.readNetFromCaffe(
    prototxt=data_path.CONFIG_MODEL_PATH,
    caffeModel=data_path.FEATURE_DETECTION_MODEL_PATH,
)

imgIndividual = cv2.imread(data_path.DATA_PATH + "inputs/face.jpg")
imgFamily = cv2.imread(data_path.DATA_PATH + "inputs/family.jpg")

img = imgFamily

faces, confidenceList = face_detection.detectFaces(img, net)
imgDetectedFaces = img.copy()

for i in range(len(faces)):
    cv2.rectangle(imgDetectedFaces, faces[i], (255, 0, 255), 3)
    label = "Confidence: %0.4f" % confidenceList[i]
    labelSize, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(
        imgDetectedFaces,
        label,
        (faces[i][0], faces[i][1] - labelSize[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

cv2.imshow("Detected Faces", imgDetectedFaces)

facemarksDetector = cv2.face.createFacemarkLBF()
facemarksDetector.loadModel(data_path.LBF_MODEL_PATH)
ret, facemarks = facemarksDetector.fit(img, faces)

imgDetectedFeatures = img.copy()

for face in facemarks:
    facemarksList = face[0].astype(int)
    for i in range(len(facemarksList)):
        cv2.circle(imgDetectedFeatures, facemarksList[i], 2, (0, 255, 0), -1)
        cv2.putText(
            imgDetectedFeatures,
            "{}".format(i),
            facemarksList[i],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

# for facemark in facemarks:
#     cv2.face.drawFacemarks(imgDetectedFeatures, facemark, (255, 255, 0))


cv2.imshow("Detected Features", imgDetectedFeatures)
cv2.waitKey(0)
cv2.destroyAllWindows()
