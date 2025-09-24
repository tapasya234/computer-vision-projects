import cv2
import glob
import numpy as np
from data_path import DATA_PATH

# Load the cascade face classifier from XML
faceCascadeClassifier = cv2.CascadeClassifier(
    DATA_PATH + "models/haarcascade_frontalface_default.xml"
)
faceNeighboursMax = 10
faceNeightboursStep = 1

# Load the cascade smile classifier from XML
# TODO: Only Load this when detecting smiles.
smileCascadeClassifier = cv2.CascadeClassifier(
    DATA_PATH + "models/haarcascade_smile.xml"
)
smileNeighboursMax = 90
smileNeightboursStep = 10


def detectFace(img):
    count = 1
    imgGS = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgHeight, imgWidth = img.shape[:2]
    for neighbour in range(1, faceNeighboursMax, faceNeightboursStep):
        faces = faceCascadeClassifier.detectMultiScale(imgGS, 1.2, neighbour)
        imgCopy = np.copy(img)
        if len(faces) == 0:
            break

        for x, y, width, height in faces:
            cv2.rectangle(
                imgCopy,
                (x, y),
                (x + width, y + height),
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            imgCopy,
            f"# Neighbours: {neighbour}",
            (int(imgWidth * 0.3), int(imgHeight * 0.1)),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            5,
        )
        count += 1
        cv2.imshow("Count: {}".format(count), imgCopy)
    cv2.waitKey(0)


def detectSmile(img):
    imgGS = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascadeClassifier.detectMultiScale(imgGS, 1.4, 5)
    if len(faces) == 0:
        return

    for x, y, width, height in faces:
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 4)

        faceROIGS = imgGS[y : y + height, x : x + width]

    count = 1
    imgHeight, imgWidth = img.shape[:2]
    for neighbour in range(1, smileNeighboursMax, smileNeightboursStep):
        smiles = smileCascadeClassifier.detectMultiScale(faceROIGS, 1.5, neighbour)
        if len(smiles) == 0:
            break

        imgCopy = np.copy(img)
        faceROICopy = imgCopy[y : y + height, x : x + width]
        for smileX, smileY, smileWidth, smileHeight in smiles:
            cv2.rectangle(
                faceROICopy,
                (smileX, smileY),
                (smileX + smileWidth, smileY + smileHeight),
                (0, 255, 0),
                2,
            )
        cv2.putText(
            imgCopy,
            f"# Neighbours: {neighbour}",
            (int(imgWidth * 0.3), int(imgHeight * 0.1)),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            5,
        )
        count += 1
        cv2.imshow("Count: {}".format(count), imgCopy)
    cv2.waitKey(0)


for imgPath in glob.glob(DATA_PATH + "testData/*.jpg"):
    print(imgPath)
    img = cv2.imread(imgPath)
    # detectFace(img)
    detectSmile(img)

cv2.destroyAllWindows()
