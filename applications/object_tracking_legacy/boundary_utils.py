import cv2


def drawBoundingBox(frame, bbox, boundaryColour=(255, 255, 0)):
    cv2.rectangle(
        frame,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[0] + bbox[2]), int(bbox[3] + bbox[1])),
        boundaryColour,
        2,
        cv2.LINE_AA,
    )
