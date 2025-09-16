import os

DATA_PATH = os.path.join(
    os.getcwd(), "python3/compVision/neural_network/text_detection_recognition/"
)

EAST_MODEL_PATH = DATA_PATH + "models/frozen_east_text_detection.pb"
DB_18_MODEL_PATH = DATA_PATH + "models/DB_TD500_resnet18.onnx"
DB_50_MODEL_PATH = DATA_PATH + "models/DB_TD500_resnet50.onnx"

VOCAB_PATH = DATA_PATH + "models/alphabet_94.txt"

CRNN_NET_OBJ_PATH = DATA_PATH + "models/crnn_cs.onnx"
