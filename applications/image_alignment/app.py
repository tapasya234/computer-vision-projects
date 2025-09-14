import cv2
from data_path import DATA_PATH
import numpy
from matplotlib import pyplot

# The multi-step process includes the following steps:
# 1. Keypoint detection and feature extraction (in both images)
# 1. Keypoint matching (between the two images)
# 1. Computing the homography that relates the two images
# 1. Using the homograhy to warp the perspective of the original image

imgForm = cv2.imread(DATA_PATH + "form.jpg")
imgScannedForm = cv2.imread(DATA_PATH + "scanned-form.jpg")

# cv2.imshow("Form", imgForm)
# cv2.imshow("Scanned", imgScannedForm)

# Detect Keypoints using ORB
orb = cv2.ORB.create(600)
imgFormGs = cv2.cvtColor(imgForm, cv2.COLOR_BGR2GRAY)
keypointsForm, descriptorsForm = orb.detectAndCompute(image=imgFormGs, mask=None)

imgScannedGs = cv2.cvtColor(imgScannedForm, cv2.COLOR_BGR2GRAY)
keypointsScanned, descriptorsScanned = orb.detectAndCompute(
    image=imgScannedGs, mask=None
)

print(len(keypointsForm))
print(len(keypointsScanned))
imgFormKeypoints = cv2.drawKeypoints(
    image=imgForm,
    keypoints=keypointsForm,
    outImage=None,
    color=(255, 0, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)
imgScannedKeypoints = cv2.drawKeypoints(
    image=imgScannedForm,
    keypoints=keypointsScanned,
    outImage=None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)
cv2.imshow("Form Keypoints", imgFormKeypoints)
cv2.imshow("Scanned Keypoints", imgScannedKeypoints)

print("Template Form")
print(
    "Keypoint Angle: {}\nKeypoint Size: {}\nKeypint point: {}".format(
        keypointsForm[0].angle,
        keypointsForm[0].size,
        keypointsForm[0].pt,
    )
)

print("Descriptor: ", descriptorsForm[0])

print("\n\nScanned Form")
print(
    "Keypoint Angle: {}\nKeypoint Size: {}\nKeypint point: {}".format(
        keypointsScanned[0].angle,
        keypointsScanned[0].size,
        keypointsScanned[0].pt,
    )
)
print("Descriptor: ", descriptorsScanned[0])

# pyplot.plot(descriptorsForm[0])
# pyplot.plot(descriptorsScanned[0])
# pyplot.title("Two Random Descriptor Vectors")

# Match Keypoints
matcher = cv2.DescriptorMatcher.create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptorsForm, descriptorsScanned, mask=None)

matchesSorted = sorted(matches, key=lambda x: x.distance, reverse=False)
goodMatchedCount = int(len(matchesSorted) * 0.1)
matches = matchesSorted[:goodMatchedCount]

queryIndex = matches[0].queryIdx
trainIndex = matches[0].trainIdx
print("Template Form")
print(
    "Keypoint Angle: {}\nKeypoint Size: {}\nKeypint point: {}".format(
        keypointsForm[queryIndex].angle,
        keypointsForm[queryIndex].size,
        keypointsForm[queryIndex].pt,
    )
)

print("Descriptor: ", descriptorsForm[queryIndex])

print("\n\nScanned Form")
print(
    "Keypoint Angle: {}\nKeypoint Size: {}\nKeypint point: {}".format(
        keypointsScanned[trainIndex].angle,
        keypointsScanned[trainIndex].size,
        keypointsScanned[trainIndex].pt,
    )
)
print("Descriptor: ", descriptorsScanned[trainIndex])

pyplot.plot(descriptorsForm[queryIndex])
pyplot.plot(descriptorsScanned[trainIndex])
pyplot.title("Best Keypoint Match Descriptor Vector")

pyplot.show()

imgMatches = cv2.drawMatches(
    imgForm, keypointsForm, imgScannedForm, keypointsScanned, matches, None
)
cv2.imshow("Matches", imgMatches)

pointsForm = numpy.zeros((len(matches), 2), dtype=numpy.float32)
pointsScanned = numpy.zeros((len(matches), 2), dtype=numpy.float32)

for i, match in enumerate(matches):
    pointsForm[i:] = keypointsForm[match.queryIdx].pt
    pointsScanned[i:] = keypointsScanned[match.trainIdx].pt
h, mask = cv2.findHomography(pointsScanned, pointsForm, method=cv2.RANSAC)

imgScannedProper = cv2.warpPerspective(
    imgScannedForm, h, (imgForm.shape[1], imgForm.shape[0])
)
cv2.imshow("Updated", imgScannedProper)
cv2.waitKey(0)
cv2.destroyAllWindows()
