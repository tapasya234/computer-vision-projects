import cv2
import numpy as np


# Converts the hog descriptors from array to Mat type. This is required for giving it to the SVM training
def prepareDataForSVM(data):
    featureVectorLength = len(data[0])
    features = np.float32(data).reshape(-1, featureVectorLength)
    return features


# The SVM model is initialized and some of its parameters are set
def svmInit(C, gamma) -> cv2.ml.SVM:
    model = cv2.ml.SVM().create()
    model.setC(C)
    model.setGamma(gamma)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setType(cv2.ml.SVM_C_SVC)

    return model


# The SVM classifier is trained using the function.
# The data for training needs to be given in the specific format as given below. The model is saved to a yml file.
def svmTrain(model: cv2.ml.SVM, trainingData, responses) -> cv2.ml.SVM:
    model.train(trainingData, cv2.ml.ROW_SAMPLE, responses)
    return model


def svmPredict(model: cv2.ml.SVM, data):
    return model.predict(data)[1].ravel()


# Predicts the classification type for the provided data and returns the calculated accuracy
def svmEvaluate(model: cv2.ml.SVM, data, labels):
    predictions = svmPredict(model, data)
    accurancy = (predictions == labels).mean()
    print("Accuracy Percent: %0.2f %%" % (accurancy * 100))
    return accurancy
