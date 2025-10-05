import cv2
import numpy as np
import sys

from data_path import DATA_PATH
import tracker_type
import banner_utils
import boundary_utils


cap = cv2.VideoCapture(DATA_PATH + "surfing.mp4")
if not cap.isOpened():
    print("Could not find/open video")
    sys.exit()

hasFrame, frame = cap.read()
if not hasFrame:
    print("Failure to read video")
    sys.exit()

frameCopy = banner_utils.addBanner(frame)
banner_utils.addText(frameCopy, "Select ROI", (150, 50))
bbox = None
if bbox is None:
    bbox = cv2.selectROI(frameCopy, True)

cv2.destroyAllWindows()

trackType = tracker_type.TRACKER_TYPE_TLD
tracker = tracker_type.createTrackerInstance(trackType)
tracker.init(frameCopy, bbox)

outputFileName = "tracked_surfing_" + trackType + ".mp4"
frameHeight, frameWidth = frameCopy.shape[:2]
fps = int(cap.get(cv2.CAP_PROP_FPS))

videoWriter = cv2.VideoWriter(
    DATA_PATH + outputFileName,
    cv2.VideoWriter.fourcc(*"mp4v"),
    20,
    (frameWidth, frameHeight),
)

windowName = "Object Tracking"
while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        break

    updatedFrame = banner_utils.addBanner(frame)
    startTimer = cv2.getTickCount()
    ok, bbox = tracker.update(updatedFrame)

    if ok:
        newFps = cv2.getTickFrequency() / (cv2.getTickCount() - startTimer)
        banner_utils.addText(
            updatedFrame,
            trackType + " Tracker. FPS: " + str(int(newFps)),
            location=(150, 50),
        )
        boundary_utils.drawBoundingBox(updatedFrame, bbox)
        cv2.waitKey(1)
    else:
        banner_utils.addText(
            updatedFrame, "Tracking failure detected", fontColour=(0, 0, 255)
        )

    cv2.imshow(windowName, updatedFrame)
    videoWriter.write(updatedFrame)

print("Finished creating new video: ", outputFileName)
cap.release()
videoWriter.release()

cv2.destroyAllWindows()
