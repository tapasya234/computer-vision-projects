import os

DATA_PATH = os.path.join(
    os.getcwd(), "python3/compVision/applications/social_distancing_detector/"
)

MODEL_FILE = DATA_PATH + "models/MobileNetSSD_deploy.caffemodel"
CONFIG_FILE = DATA_PATH + "models/MobileNetSSD_deploy.prototxt"
