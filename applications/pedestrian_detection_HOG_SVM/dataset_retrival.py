import cv2
import os


def getImagePaths(directoryPath, imageExtensions):
    """
    getImagePaths returns image paths for the images in the provided directory
    whose extensions are present in the provided `imageExtensions`.

    :param directoryPath: Path to the directory, provided as string.
    :param imageExtensions: List of acceptable image extensions
    """

    imagePaths = []
    for file in os.listdir(directoryPath):
        if os.path.splitext(file)[1] in imageExtensions:
            imagePaths.append(os.path.join(directoryPath, file))

    return imagePaths


def getDatasets(directoryPath, classLabel):
    """
    getDatasets returns the list images in the provided directory
    along with a list of the respective labels.

    :param directoryPath: Path to the directory, provided as a string
    :param classLabel: Int value to assign a label for each image
    """

    images = []
    labels = []

    imagePaths = getImagePaths(directoryPath, [".jpg", ".png", ".jpeg"])

    for imagePath in imagePaths:
        images.append(cv2.imread(imagePath, cv2.IMREAD_COLOR))
        labels.append(classLabel)

    return images, labels
