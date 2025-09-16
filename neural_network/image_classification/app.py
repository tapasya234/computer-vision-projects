import cv2
import numpy
import glob
from data_path import DATA_PATH

# Read the ImageNet class names.
with open(DATA_PATH + "input/classification_classes_ILSVRC2012.txt", "r") as f:
    imageNetNames = f.read().split("\n")

print(len(imageNetNames), imageNetNames[1])
imageNetNames = imageNetNames[:-1]

print(len(imageNetNames), imageNetNames[1])

configFile = DATA_PATH + "models/DenseNet_121.prototxt"
modelFile = DATA_PATH + "models/DenseNet_121.caffemodel"

model = cv2.dnn.readNet(model=modelFile, config=configFile, framework="Caffe")


def classify(img):
    image = img.copy()
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=0.01,
        size=(224, 224),
        mean=(104, 117, 124),
        # swapRB=False,
        # crop=False,
    )

    model.setInput(blob)

    outputs = model.forward()
    finalOutputs = outputs[0]

    # Make all the outputs 1D.
    finalOutputs = finalOutputs.reshape(1000, 1)

    # ind = numpy.argpartition(finalOutputs, -4)[-4:]
    # print(finalOutputs[ind])
    # Get the class label
    # finalOutputs = numpy.sort(finalOutputs, axis=None)
    # print(finalOutputs[:10])
    labelID = numpy.argmax(finalOutputs)
    
    # Convert score to probabilites
    probability = numpy.exp(finalOutputs) / numpy.sum(numpy.exp(finalOutputs))
    # print(probability[:10])

    # Get final highest probablity
    finalProbability = numpy.max(probability) * 100

    # Map the max confidence to the class label names
    identifiedClass = imageNetNames[labelID]
    print(identifiedClass, finalProbability)
    return f"{identifiedClass}, {finalProbability:.3f}%"


images = []
imagesClass = []

for path in glob.glob(DATA_PATH + "input/*.jpg"):
    img = cv2.imread(path)
    images.append(img)
    print("Classifying image ", path)
    imagesClass.append(classify(img))

# path = DATA_PATH + "input/image5.jpg"
# img = cv2.imread(path)
# images.append(img)
# print("Classifying image ", path)
# imagesClass.append(classify(img))

for i in range(len(images)):
    cv2.imshow(imagesClass[i], images[i])
    cv2.waitKey(0)

cv2.destroyAllWindows()
