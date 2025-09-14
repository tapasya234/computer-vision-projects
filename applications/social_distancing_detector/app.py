import cv2
import numpy as np
import data_path

SCALE_FACTOR = 0.007843
MEAN = (127.5, 127.5, 127.5)
INPUT_DIMENSIONS = 300


def detectPeople(frame, net: cv2.dnn.Net):
    # print("Detecting People...")
    detectedPeopleDetails = []
    frameHeight, frameWidth = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        SCALE_FACTOR,
        (INPUT_DIMENSIONS, INPUT_DIMENSIONS),
        MEAN,
        False,
        False,
    )
    net.setInput(blob)
    outputs = net.forward()

    for i in np.arange(0, outputs.shape[2]):
        classID = outputs[0, 0, i, 1]
        confidence = outputs[0, 0, i, 2]

        if confidence > 0.7 and classID == 15:
            box = outputs[0, 0, i, 3:7] * np.array(
                [frameWidth, frameHeight, frameWidth, frameHeight]
            )
            box = box.astype("int")

            centerX = int((box[0] + box[2]) / 2)
            centerY = int((box[1] + box[3]) / 2)

            detectedPeopleDetails.append((confidence, box, (centerX, centerY)))
    return detectedPeopleDetails


def calcEuclideanDist(A: np.array, B: np.array):
    part1 = np.sum(A**2, axis=1)[:, np.newaxis]
    part2 = np.sum(B**2, axis=1)
    part3 = -2 * np.dot(A, B.T)
    return np.round(np.sqrt(part1 + part2 + part3), 2)


def detectViolations(detectedPeopleResults):
    # print(
    #     "Detected People: {}, Detecting Violations...".format(
    #         len(detectedPeopleResults)
    #     )
    # )
    violations = set()
    if len(detectedPeopleResults) >= 2:
        violations = set()
        factor = 1.2

        boxesWidth = np.array(
            [abs(int(r[1][2] - r[1][0])) for r in detectedPeopleResults]
        )
        centroids = np.array([r[2] for r in detectedPeopleResults])
        distanceMatrix = calcEuclideanDist(centroids, centroids)

        for row in range(distanceMatrix.shape[0]):
            for col in range(row + 1, distanceMatrix.shape[1]):
                refDistance = int(factor * min(boxesWidth[row], boxesWidth[col]))

                if distanceMatrix[row][col] < refDistance:
                    violations.add(row)
                    violations.add(col)
    return violations


def drawBoundingBoxes(img, detectedPeopleResults, violations):
    # print("Drawing bounding boxes...")
    frame = img.copy()
    for index, (prob, boundingBox, centroid) in enumerate(detectedPeopleResults):
        x1, y1, x2, y2 = boundingBox
        colour = (0, 0, 255) if index in violations else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)

        cv2.putText(
            frame,
            "Not Safe" if index in violations else "Safe",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            colour,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Num Violations: {len(violations)}",
            (10, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


net = cv2.dnn.readNetFromCaffe(
    prototxt=data_path.CONFIG_FILE,
    caffeModel=data_path.MODEL_FILE,
)

# img = cv2.imread(data_path.DATA_PATH + "input/input_frame.png")
# detectedPeopleResults = detectPeople(img, net)
# violations = detectViolations(detectedPeopleResults)
# updatedImg = drawBoundingBoxes(img, detectedPeopleResults, violations)
# cv2.imshow("Detected Violations", updatedImg)
# cv2.waitKey(0)

cap = cv2.VideoCapture(data_path.DATA_PATH + "input/input2.mp4")
windowName = "Detected Violations"
while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        break
    detectedPeopleResults = detectPeople(frame, net)
    violations = detectViolations(detectedPeopleResults)
    updatedImg = drawBoundingBoxes(frame, detectedPeopleResults, violations)
    cv2.imshow(windowName, updatedImg)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
