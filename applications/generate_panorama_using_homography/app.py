import cv2
import numpy as np
from data_path import DATA_PATH

# Creating a panorama for 2 images consists of the following steps:
#  - Find Keypoints and Descriptors for both images.
#  - Find Corresponding points by matching their Descriptors.
#  - Align second image with respect to first image using Homography.
#  - Warp the second image using Perspective Transformation.
#  - Combine the first image with the warped image to get the Panorama.


# This approach has some challenges:
# There might be visible seams at the boundary of the two images. This is because of the variation in lighting / exposure between the two images.
# The lighting variation might also require some color correction as the two images might not blend well after stitching.
# Difficult to extend to multiple images.

MAX_FEATURES = 500
GOOD_MATCHES_PERCENT = 0.5

# FIND KEYPOINTS AND DESCRIPTORS FOR EACH IMAGE
# Read 1st image of the scene, reference image
img1 = cv2.imread(DATA_PATH + "scene/scene1.jpg")
img1GS = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Read 3rd image of the scene, image to align with reference image
img2 = cv2.imread(DATA_PATH + "scene/scene2.jpg")
img2GS = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB().create()
keypoints1, descriptors1 = orb.detectAndCompute(img1GS, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2GS, None)

img1Keypoints = cv2.drawKeypoints(
    img1,
    keypoints1,
    outImage=None,
    color=(255, 0, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
)
cv2.imshow("Img1 Keypoints", img1Keypoints)

img2Keypoints = cv2.drawKeypoints(
    img2,
    keypoints2,
    outImage=None,
    color=(0, 0, 255),
    flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
)
cv2.imshow("Img2 Keypoints", img2Keypoints)

# FIND MATCHING CORESPONDING POINTS
matcher = cv2.DescriptorMatcher().create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2, None)

matches = sorted(matches, key=lambda x: x.distance, reverse=False)
goodMatchesCount = int(len(matches) * GOOD_MATCHES_PERCENT)
matches = matches[:goodMatchesCount]

imgMatches = cv2.drawMatches(
    img1,
    keypoints1,
    img2,
    keypoints2,
    outImg=None,
    matches1to2=matches,
)
cv2.imshow("Features Matching", imgMatches)

# IMAGE ALIGNMENT USING HOMOGRAPHY
# After matching is done, the output ( matches ) has the following attributes :
# `matches.distance` - Distance between descriptors. Should be lower for better match.
# `matches.trainIdx` - Index of the descriptor in train descriptors
# `matches.queryIdx` - Index of the descriptor in query descriptors
# `matches.imgIdx` - Index of the train image.
# To simplify things, the `queryIdx` corresponds to points in image1 and `trainIdx` corresponds to points in image2.

points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

# WARP IMAGE
img1Height, img1Width = img1.shape[:2]
img2Height, img2Width = img2.shape[:2]
print(img1.shape[:2], img2.shape[:2])

img2Aligned = cv2.warpPerspective(img2, h, ((img2Width + img1Width), img1Height))
cv2.imshow("Aligned scene", img2Aligned)

# STITCH IMAGE
stitchedImg = np.copy(img2Aligned)
stitchedImg[0:img1Height, 0:img1Width] = img1
cv2.imshow("Badly Stitched scene", stitchedImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
