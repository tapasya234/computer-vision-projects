import cv2
import numpy as np


def addBanner(frame, heightPercentage=0.08, bannerColour=(0, 0, 0)):
    bannerHeight = int(heightPercentage * frame.shape[0])
    newFrame = np.zeros(
        (bannerHeight + frame.shape[0], frame.shape[1], 3), dtype=np.uint8
    )
    if bannerColour != (0, 0, 0):
        newFrame[:bannerHeight, :, :] = bannerColour
    newFrame[bannerHeight:, :, :] = frame
    return newFrame


def addText(
    frame,
    text,
    location=(50, 25),
    fontScale=2,
    fontThickness=2,
    fontColour=(0, 255, 0),
):
    cv2.putText(
        frame,
        text,
        location,
        cv2.FONT_HERSHEY_PLAIN,
        fontScale,
        fontColour,
        fontThickness,
        cv2.LINE_AA,
    )
