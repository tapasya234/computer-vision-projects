import os

DATA_PATH = os.path.join(
    os.getcwd(),
    "python3/computer-vision-projects/assignments/project_football_detection_tracking/",
)

CONFIG_PATH = DATA_PATH + "models/yolov4-tiny.cfg"
MODEL_PATH = DATA_PATH + "models/yolov4-tiny.weights"

CLASSES_PATH = DATA_PATH + "models/coco.names"
