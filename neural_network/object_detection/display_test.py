import cv2

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_THICKNESS = 2


def displayText(img, text, x, y):
    textSize, baseline = cv2.getTextSize(text, FONT_FACE, FONT_SCALE, FONT_THICKNESS)

    cv2.rectangle(
        img,
        (x, y),
        (x + textSize[0], y + textSize[1] + baseline),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        img,
        text,
        (x, y + textSize[1]),
        FONT_FACE,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )
