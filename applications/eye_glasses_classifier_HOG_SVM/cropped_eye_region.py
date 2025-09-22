import cv2
from data_path import DATA_PATH

faceCascade = cv2.CascadeClassifier(
    DATA_PATH + "models/haarcascade_frontalface_default.xml"
)


def getCroppedEyeRegion(targetImage):
    imgGray = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)
    if faces is None or len(faces) == 0:
        return None

    faceX, faceY, faceWidth, faceHeight = faces[0]

    faceROI = targetImage[faceY : faceY + faceHeight, faceX : faceX + faceWidth]

    # Apply a heuristic formula for getting the eye region from face
    eyesTop = int(1 / 6.0 * faceHeight)
    eyesBottom = int(3 / 6.0 * faceHeight)
    # print("Eye Height between: {}, {}".format(eyesTop, eyesBottom))

    eyesROI = faceROI[eyesTop:eyesBottom, :]

    return cv2.resize(eyesROI, (96, 32), interpolation=cv2.INTER_CUBIC)
