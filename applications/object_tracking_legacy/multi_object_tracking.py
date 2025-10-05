import cv2
import cv2.legacy
from random import randint
import sys

from data_path import DATA_PATH
import banner_utils
import tracker_type
import boundary_utils

cap = cv2.VideoCapture(DATA_PATH + "cycle.mp4")
if not cap.isOpened():
    print("Could not find/open video")
    sys.exit()

hasFrame, frame = cap.read()
if not hasFrame:
    print("Failure to read video")
    sys.exit()

boundaryBoxes = []
boundaryColours = []
while True:
    # draw bounding boxes over objects
    # selectROI's default behaviour is to draw box starting from the center

    bbox = cv2.selectROI("MultiTracker", frame)
    boundaryBoxes.append(bbox)
    boundaryColours.append((randint(64, 255), randint(64, 255), randint(64, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    if k == 113:
        break

print("Selected bounding boxes {}".format(boundaryBoxes))

multiTracker = cv2.legacy.MultiTracker().create()
trackerType = tracker_type.TRACKER_TYPE_CSRT
for bbox in boundaryBoxes:
    multiTracker.add(tracker_type.createTrackerInstance(trackerType), frame, bbox)
cv2.destroyAllWindows()


windowName = "Object Tracking"
while cap.isOpened():
    hasFrame, frame = cap.read()
    if not hasFrame:
        break

    updatedFrame = banner_utils.addBanner(frame)
    startTimer = cv2.getTickCount()
    success, boxes = multiTracker.update(updatedFrame)

    if success:
        newFps = cv2.getTickFrequency() / (cv2.getTickCount() - startTimer)
        for i, boundaryBox in enumerate(boxes):
            banner_utils.addText(
                updatedFrame,
                trackerType + " Tracker. FPS: " + str(int(newFps)),
                location=(150, 50),
            )
            boundary_utils.drawBoundingBox(
                updatedFrame, boundaryBox, boundaryColours[i]
            )
            cv2.waitKey(1)
    else:
        banner_utils.addText(
            updatedFrame, "Tracking failure detected", fontColour=(0, 0, 255)
        )

    cv2.imshow(windowName, updatedFrame)

cv2.destroyAllWindows()
