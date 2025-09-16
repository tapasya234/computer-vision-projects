import cv2
import data_path
import numpy

cap = cv2.VideoCapture(0)

net = cv2.dnn.readNetFromCaffe(
    prototxt=data_path.CONFIG_MODEL_PATH,
    # caffeModel=data_path.FACE_DETECTION_MODEL_PATH,
    caffeModel=data_path.FEATURE_DETECTION_MODEL_PATH,
)


def blurFace(face, factor=3):
    faceHeight, faceWidth = face.shape[:2]

    if factor < 1:
        factor = 1
    elif factor > 5:
        factor = 5

    kernelWidth = int(faceWidth / factor)
    kernelHeight = int(faceHeight / factor)

    if kernelWidth % 2 == 0:
        kernelWidth += 1
    if kernelHeight % 2 == 0:
        kernelHeight += 1

    return cv2.GaussianBlur(face, (kernelWidth, kernelHeight), 0, 0)


while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        break

    frameHeight, frameWidth = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104, 112, 124),
        swapRB=False,
        crop=False,
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # print(confidence)
        if confidence > 0.5:
            # print(detections[0, 0, i])
            box = detections[0, 0, i, 3:7] * numpy.array(
                [
                    frameWidth,
                    frameHeight,
                    frameWidth,
                    frameHeight,
                ]
            )
            x1, y1, x2, y2 = box.astype("int")

            # face = frame[y1:y2, x1:x2, :]
            # cv2.imshow("Face", face)
            # faceBlurred = cv2.(face, (7, 7))
            frame[y1:y2, x1:x2, :] = blurFace(frame[y1:y2, x1:x2, :], factor=7)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)
            label = "Confidence: %0.4f" % confidence
            labelSize, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - labelSize[1]),
                (x1 + labelSize[0], y1 + baseline),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)
            )

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == 27:
        break
    # break

cap.release()
cv2.destroyAllWindows()
