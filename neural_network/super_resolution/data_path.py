import os

DATA_PATH = os.path.join(
    os.getcwd(), "python3/compVision/neural_network/super_resolution/"
)

EAST_MODEL_PATH = DATA_PATH + "models/frozen_east_text_detection.pb"
CRNN_NET_OBJ_PATH = DATA_PATH + "models/crnn_cs.onnx"

VOCAB_PATH = DATA_PATH + "models/alphabet_94.txt"
