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

MAX_FEATURES = 1000
GOOD_MATCHES_PERCENT = 0.1

# The minimum matches count is set this low because the method is not very reliable in real world environments,
# because of changes in luminosity, white balance, focus, spatial position of the object etc.,.
MIN_MATCH_COUNT = 10

# FIND KEYPOINTS AND DESCRIPTORS FOR EACH IMAGE
# Read 1st image of the scene, reference image
imgBook = cv2.imread(DATA_PATH + "book.jpeg")
imgBookGS = cv2.cvtColor(imgBook, cv2.COLOR_BGR2GRAY)

# Read 3rd image of the scene, image to align with reference image
imgBookScene = cv2.imread(DATA_PATH + "book_scene.jpeg")
imgBookSceneGS = cv2.cvtColor(imgBookScene, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB().create(MAX_FEATURES)
keypointsBook, descriptorsBook = orb.detectAndCompute(imgBookGS, None)
keypointsBookScene, descriptorsBookScene = orb.detectAndCompute(imgBookSceneGS, None)

imgBookFeatures = cv2.drawKeypoints(
    imgBook, keypointsBook, outImage=None, color=(255, 255, 0)
)
cv2.imshow("Book Features", imgBookFeatures)

imgBookSceneFeatures = cv2.drawKeypoints(
    imgBookScene, keypointsBookScene, outImage=None, color=(0, 255, 255)
)
cv2.imshow("Bookscene Features", imgBookSceneFeatures)


# Find the homography matrix using the matches provided and find the edges of the book in the bookscene.
def detectBookInBookscene(goodMatches):
    global imgBookScene

    srcPoints = np.float32([keypointsBook[m.queryIdx].pt for m in goodMatches]).reshape(
        -1, 1, 2
    )
    destPoints = np.float32(
        [keypointsBookScene[m.trainIdx].pt for m in goodMatches]
    ).reshape(-1, 1, 2)
    h, mask = cv2.findHomography(srcPoints, destPoints, cv2.RANSAC, 5.0)

    height, width = imgBook.shape[:2]

    # Points in Book image
    ptsBook = np.float32(
        [[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]
    ).reshape(-1, 1, 2)

    # Find corresponding points in the bookscene image
    dest = cv2.perspectiveTransform(ptsBook, h)

    imgBookScene = cv2.polylines(
        imgBookScene,
        [np.int32(dest)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=5,
        lineType=cv2.LINE_AA,
    )

    return mask.ravel().tolist()


#  Matching using BRUTE-FORCE HAMMING
def matchUsingBruteForce():
    matcherBF = cv2.DescriptorMatcher().create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    )
    matchesBF = matcherBF.match(descriptorsBook, descriptorsBookScene)
    matchesBF = sorted(matchesBF, key=lambda x: x.distance, reverse=False)
    goodMatchesCount = int(len(matchesBF) * GOOD_MATCHES_PERCENT)
    goodMatches = matchesBF[:goodMatchesCount]

    mask = detectBookInBookscene(goodMatches)

    imgMatchesBF = cv2.drawMatches(
        imgBook,
        keypointsBook,
        imgBookScene,
        keypointsBookScene,
        matches1to2=goodMatches,
        outImg=None,
    )
    cv2.imshow("Brute Force Matches", imgMatchesBF)


# Matching using FLANN
# Find good mathes using Lowe's Ratio test.
# Lowe's Ratio Test - "Correct matches need to have the closest neighbour significantly
# closer than the closest incorrect match to achieve eliable matching."
# In other words, the distance associated with the best match should be much
# better than the distance associated with the second best match.
def matchUsingFlann():
    FLANN_INDEX_KDTREE = 1
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParama = dict(checks=50)
    matcherFlann = cv2.FlannBasedMatcher(
        indexParams=indexParams, searchParams=searchParama
    )
    matchesFlann = matcherFlann.knnMatch(
        np.float32(descriptorsBook), np.float32(descriptorsBookScene), k=2
    )

    goodMatches = []
    for m, n in matchesFlann:
        if m.distance < 0.9 * n.distance:
            goodMatches.append(m)

    if len(goodMatches) < MIN_MATCH_COUNT:
        print("Not enough matches were found. {}/{}", len(goodMatches), MIN_MATCH_COUNT)
        matchesMask = None
    else:
        matchesMask = detectBookInBookscene(goodMatches)

    draw_params = dict(
        matchColor=(255, 0, 0),
        singlePointColor=None,
        matchesMask=matchesMask,  # Draw only inliers
        flags=2,
    )
    imgMatchesFlann = cv2.drawMatches(
        imgBook,
        keypointsBook,
        imgBookScene,
        keypointsBookScene,
        goodMatches,
        None,
        **draw_params
    )
    cv2.imshow("FLANN Matches", imgMatchesFlann)


matchUsingBruteForce()
matchUsingFlann()

cv2.waitKey(0)
cv2.destroyAllWindows()
