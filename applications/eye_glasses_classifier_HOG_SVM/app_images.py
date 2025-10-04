import cv2
import numpy as np
from data_path import DATA_PATH
from cropped_eye_region import getCroppedEyeRegion
import data_preperation
import svm_classifier
import hog_feature_descriptor

classLabelNoGlasses = 1
classLabelGlasses = 0
predictionsLabel = {classLabelNoGlasses: "No Glasses", classLabelGlasses: "Glasses"}


def trainModel(trainFeatures, trainLabels) -> cv2.ml.SVM:
    # 3. Train SVM Model
    # Specify the parameters for SVM Classifier and train the model using the training features obtained above.
    model = svm_classifier.svmInit(C=2.5, gamma=0.02)
    model = svm_classifier.svmTrain(model, trainFeatures, trainLabels)
    model.save(DATA_PATH + "models/eyeGlassesClassifierModel.yml")
    return model


# 1. Compute train and test data
trainDataGlasses, trainLabelsGlasses, testDataGlasses, testLabelsGlasses = (
    data_preperation.getTrainTestData(
        data_preperation.pathGlasses, classLabelGlasses, 0.2
    )
)

trainDataNoGlasses, trainLabelsNoGlasses, testDataNoGlasses, testLabelsNoGlasses = (
    data_preperation.getTrainTestData(
        data_preperation.pathNoGlasses, classLabelNoGlasses, 0.2
    )
)

trainImages = np.concatenate(
    (np.array(trainDataNoGlasses), np.array(trainDataGlasses)),
    axis=0,
)
trainLabels = np.concatenate(
    (np.array(trainLabelsNoGlasses), np.array(trainLabelsGlasses)),
    axis=0,
)

testImages = np.concatenate(
    (np.array(testDataNoGlasses), np.array(testDataGlasses)),
    axis=0,
)
testLabels = np.concatenate(
    (np.array(testLabelsNoGlasses), np.array(testLabelsGlasses)),
    axis=0,
)

# 2. Compute Features
trainHOG = hog_feature_descriptor.computerHOG(hog_feature_descriptor.hog, trainImages)
testHOG = hog_feature_descriptor.computerHOG(hog_feature_descriptor.hog, testImages)

# Convert hog data into features recognized by SVM model
trainFeatures = svm_classifier.prepareDataForSVM(trainHOG)
testFeatures = svm_classifier.prepareDataForSVM(testHOG)


# model = trainModel(trainFeatures, trainLabels)

# 4. Evaluate the model and check the accuracy
# Load the provided model
model = cv2.ml.SVM().load(DATA_PATH + "models/eyeGlassesClassifierModel.yml")
accuracy = svm_classifier.svmEvaluate(model, testFeatures, testLabels)

# 5. Perform the classification on a specific image
wrongPredictionsFileNames = []
faceNotFoundFileNames = []


def predictAndDisplayImage(
    imageFilePath: str, hog: cv2.HOGDescriptor, model: cv2.ml.SVM
):
    img = cv2.imread(imageFilePath)
    roi = getCroppedEyeRegion(img)

    fileName = imageFilePath.split("/")[-1]
    # cv2.imwrite(DATA_PATH + "glassesDataset/" + fileName, roi)
    imgHeight, imgWidth = img.shape[:2]
    if roi is None:
        cv2.putText(
            img,
            "Face not found",
            (int(imgWidth * 0.3), int(imgHeight * 0.9)),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        faceNotFoundFileNames.append(fileName)
    else:
        testHOG = hog_feature_descriptor.computerHOG(
            hog_feature_descriptor.hog, np.array([roi])
        )
        testFeatures = svm_classifier.prepareDataForSVM(testHOG)
        predictionClass = svm_classifier.svmPredict(model, testFeatures)[0]

        actualClass = fileName.startswith("no")
        if predictionClass != actualClass:
            wrongPredictionsFileNames.append(fileName)

        cv2.putText(
            img,
            f"Prediction: {predictionsLabel[int(predictionClass)]}",
            (int(imgWidth * 0.3), int(imgHeight * 0.9)),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Actual: {predictionsLabel[int(actualClass)]}",
            (int(imgWidth * 0.3), int(imgHeight * 0.95)),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imshow(fileName, img)
    cv2.waitKey(0)


testInputDir, testFileNames = data_preperation.retrieveImageFilenamesFromDataset(
    DATA_PATH + "glassesDataset/testData_fullFace"
)

for imgFileName in testFileNames:
    predictAndDisplayImage(
        testInputDir + "/" + imgFileName,
        hog_feature_descriptor.hog,
        model,
    )

print(
    "Wrong Prediction Glasses Count: {} List: {}".format(
        len(wrongPredictionsFileNames), wrongPredictionsFileNames
    )
)
print(
    "Face Not Found Error Count: {} List: {}".format(
        len(faceNotFoundFileNames), faceNotFoundFileNames
    )
)
cv2.destroyAllWindows()
