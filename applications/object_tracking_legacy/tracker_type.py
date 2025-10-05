import cv2.legacy

TRACKER_TYPE_BOOSTING = "Boosting"
TRACKER_TYPE_CSRT = "CSRT"
TRACKER_TYPE_KCF = "KCF"
TRACKER_TYPE_MEDIANFLOW = "MedianFlow"
TRACKER_TYPE_MIL = "MIL"  # Multiple Instance Learning
TRACKER_TYPE_MOSSE = "MOSSE"
TRACKER_TYPE_TLD = "TLD"


def createTrackerInstance(trackerType):
    if trackerType == TRACKER_TYPE_BOOSTING:
        return cv2.legacy.TrackerBoosting().create()
    if trackerType == TRACKER_TYPE_CSRT:
        return cv2.legacy.TrackerCSRT().create()
    if trackerType == TRACKER_TYPE_KCF:
        return cv2.legacy.TrackerKCF().create()
    if trackerType == TRACKER_TYPE_MEDIANFLOW:
        return cv2.legacy.TrackerMedianFlow().create()
    if trackerType == TRACKER_TYPE_MIL:
        return cv2.legacy.TrackerMIL().create()
    if trackerType == TRACKER_TYPE_MOSSE:
        return cv2.legacy.TrackerMOSSE().create()
    if trackerType == TRACKER_TYPE_TLD:
        return cv2.legacy.TrackerTLD().create()
