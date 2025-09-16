import os

DATA_PATH = os.path.join(
    os.getcwd(), "python3/compVision/neural_network/face_detection/"
)
FACE_DETECTION_MODEL_PATH = (
    DATA_PATH + "models/res10_300x300_ssd_iter_140000.caffemodel"
)
FEATURE_DETECTION_MODEL_PATH = (
    DATA_PATH + "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
)
CONFIG_MODEL_PATH = DATA_PATH + "models/deploy.prototxt"
LBF_MODEL_PATH = DATA_PATH + "models/lbfmodel.yaml"
