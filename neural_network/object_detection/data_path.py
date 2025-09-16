import os

DATA_PATH = os.path.join(
    os.getcwd(), "python3/compVision/neural_network/object_detection/"
)
SSD_MODEL_PATH = DATA_PATH + "models/ssd_mobilenet_frozen_inference_graph.pb"
SSD_CONFIG_PATH = DATA_PATH + "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
SSD_CLASS_PATH = DATA_PATH + "models/coco_class_labels.txt"

YOLO4_CLASS_PATH = DATA_PATH + "models/coco.names"
YOLO4_CONFIG_PATH = DATA_PATH + "models/yolov4.cfg"
YOLO4_WEIGHTS_PATH = DATA_PATH + "models/yolov4.weights"
