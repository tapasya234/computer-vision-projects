import cv2
import numpy as np


def svmInit(C, gamma):
    """
    svmInit is used to create a basic SVM model and
    set a few paramaters associated with the model.

    :param C: Parameter for the SVM model.
    :param gamma: Parameter for the SVM model.
    """

    model = cv2.ml.SVM().create()
    model.setC(C)
    model.setGamma(gamma)
    model.setKernel(cv2.ml.SVM_LINEAR)
    model.setType(cv2.ml.SVM_C_SVC)
    model.setTermCriteria(
        (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            1000,
            1e-3,
        )
    )

    return model


def svmTrain(model: cv2.ml.SVM, trainData, trainLabels):
    model.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)


def svmPredict(model: cv2.ml.SVM, testData):
    return model.predict(testData)[1]


def svmEvaluate(model: cv2.ml.SVM, testData, actualTestLabels):
    """
    svmEvaluate calculate the accuracy of the model by  predicting the provided
    testData and comparing the predictions with the actual labels.

    :param model: The trained model for the project
    :type model: cv2.ml.SVM
    :param testData: The test data used to predict the accuracy of the model.
    :param expectedTestLabels: The labels associated with the test data.
    """

    labels = labels[:, np.newaxis]
    predictions = svmPredict(model, testData)

    correctPredictionsCount = np.sum((predictions == labels))
    wrongPredictionsValue = (predictions != labels).mean()
    print(
        "Prediction Labels -- -1: {} 1: {}".format(
            np.sum(predictions == -1),
            np.sum(predictions == 1),
        )
    )
    return correctPredictionsCount, wrongPredictionsValue * 100
