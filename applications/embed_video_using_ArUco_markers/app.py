import cv2
from data_path import DATA_PATH
import mimetypes
import numpy

mimetypes.init()

case1_imgInImg = ["Apollo-8-Launch.png", "office_markers.jpg"]
case2_imgInVideo = ["New_Zealand_Cove.jpg", "office_markers.mp4"]
case3_videoInImg = ["boys_playing.mp4", "office_markers.jpg"]
case4_videoInVideo = ["horse_race.mp4", "office_markers.mp4"]

case = case3_videoInImg

# Scale factors used to increase size of source media to cover ArUco Marker borders.
scaleFactorX = 0.08
scaleFactorY = 0.12

markerIds = [23, 25, 30, 33]

prefix = "AR_"
image = "image"
video = "video"


class MediaSpec:
    def __init__(self, src, dest):
        self.src = src
        self.dest = dest


mediaSpec = MediaSpec(case[0], case[1])
srcInput = mediaSpec.src
destInput = mediaSpec.dest

# Determine the media types for source and destination, which should be `image` or `video`
srcMime = mimetypes.guess_type(srcInput)[0]
if srcMime != None:
    srcMime = srcMime.split("/")[0]
destMime = mimetypes.guess_type(destInput)[0]
if destMime != None:
    destMime = destMime.split("/")[0]

# Read the destination image/video
if destMime == image:
    destFrame = cv2.imread(DATA_PATH + destInput)
elif destMime == video:
    destVideoCap = cv2.VideoCapture(DATA_PATH + destInput)
    fps = destVideoCap.get(cv2.CAP_PROP_FPS)
    print(destVideoCap.get(cv2.CAP_PROP_FRAME_COUNT))

# Read the source image/video
if srcMime == image:
    srcFrame = cv2.imread(DATA_PATH + srcInput)
elif srcMime == video:
    srcVideoCap = cv2.VideoCapture(DATA_PATH + srcInput)
    fps = srcVideoCap.get(cv2.CAP_PROP_FPS)
    print(srcVideoCap.get(cv2.CAP_PROP_FRAME_COUNT))


if srcMime == video or destMime == video:
    outputFileName = prefix + srcMime + "_in_" + destMime + ".avi"

    if destMime == video:
        width = round(2 * destVideoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(destVideoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        width = round(2 * destFrame.shape[1])
        height = round(destFrame.shape[0])

    videoWriter = cv2.VideoWriter(
        DATA_PATH + outputFileName,
        cv2.VideoWriter.fourcc(*"XVID"),
        fps,
        (width, height),
    )
else:
    outputFileName = prefix + "image_in_image.jpg"


def extractPoints(markerIDs, detectedIds, detectedCorners):
    index = numpy.squeeze(numpy.where(detectedIds == markerIDs[0]))
    pt0 = numpy.squeeze(detectedCorners[index[0]])[0]

    index = numpy.squeeze(numpy.where(detectedIds == markerIDs[1]))
    pt1 = numpy.squeeze(detectedCorners[index[0]])[1]

    index = numpy.squeeze(numpy.where(detectedIds == markerIDs[2]))
    pt2 = numpy.squeeze(detectedCorners[index[0]])[2]

    index = numpy.squeeze(numpy.where(detectedIds == markerIDs[3]))
    pt3 = numpy.squeeze(detectedCorners[index[0]])[3]

    return pt0, pt1, pt2, pt3


def scaleDestinationPoints(pt0, pt1, pt2, pt3, scaleFactorX=0.01, scaleFactorY=0.01):
    distX = numpy.linalg.norm(pt0 - pt1)
    distY = numpy.linalg.norm(pt0 - pt3)

    deltaX = round(scaleFactorX * distX)
    deltaY = round(scaleFactorY * distY)

    # Apply the scaling factors to the ArUco Marker reference points to make
    # the final adjustment for the destination points.
    pts = [[pt0[0] - deltaX, pt0[1] - deltaY]]
    pts = pts + [[pt1[0] + deltaX, pt1[1] - deltaY]]
    pts = pts + [[pt2[0] + deltaX, pt2[1] + deltaY]]
    pts = pts + [[pt3[0] - deltaX, pt3[1] + deltaY]]

    return pts


srcHasFrame = True
destHasFrame = True

frameCount = 0
maxFramesCount = 100
colour = (255, 255, 255)

aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

print("Processing Frame...")
windowName = "Output"
while srcHasFrame & destHasFrame:
    if destMime == video:
        destHasFrame, destFrame = destVideoCap.read()
        if not destHasFrame:
            break

    if srcMime == video:
        srcHasFrame, srcFrame = srcVideoCap.read()
        if not srcHasFrame:
            break

    corners, ids, rejected = cv2.aruco.detectMarkers(destFrame, aruco_dictionary)
    refPt0, refPt1, refPt2, refPt3 = extractPoints(markerIds, ids, corners)
    destPts = scaleDestinationPoints(
        refPt0, refPt1, refPt2, refPt3, scaleFactorX, scaleFactorY
    )

    srcPts = [
        [0, 0],
        [srcFrame.shape[1], 0],
        [srcFrame.shape[0], srcFrame.shape[1]],
        [0, srcFrame.shape[0]],
    ]

    srcPtsList = numpy.asarray(srcPts)
    destPtsList = numpy.asarray(destPts)

    h, mask = cv2.findHomography(srcPtsList, destPtsList, cv2.RANSAC)

    imgWarpped = cv2.warpPerspective(
        srcFrame, h, (destFrame.shape[1], destFrame.shape[0])
    )

    mask = numpy.zeros((destFrame.shape[0], destFrame.shape[1]), dtype=numpy.uint8)

    cv2.fillConvexPoly(mask, numpy.int32(destPtsList), (255, 255, 255), cv2.LINE_AA)
    imgWarpped = imgWarpped.astype(float)

    # print(imgWarpped.shape)
    # mask3 = numpy.zeros_like(imgWarpped)
    # for i in range(0, 3):
    #     mask3[:, :, i] = mask / 255

    mask3 = cv2.merge([mask / 255, mask / 255, mask / 255])

    maskedFrame = numpy.multiply(destFrame.astype(float), 1 - mask3)
    outputFrame = cv2.add(imgWarpped, maskedFrame)
    outputConcatenated = numpy.hstack([destFrame.astype(float), outputFrame])

    widthOutputFrame = outputConcatenated.shape[1]
    heighthOutputFrame = outputConcatenated.shape[0]
    cv2.line(
        outputConcatenated,
        (int(widthOutputFrame / 2), 0),
        (int(widthOutputFrame / 2), heighthOutputFrame),
        colour,
        8,
    )

    print(outputFileName)
    if srcMime == image and destMime == image:
        cv2.imwrite(DATA_PATH + outputFileName, outputConcatenated.astype(numpy.uint8))
        cv2.imshow(windowName, outputConcatenated.astype(numpy.uint8))
        cv2.waitKey(0)
        break

    cv2.imshow(windowName, outputConcatenated.astype(numpy.uint8))
    cv2.waitKey(0)
    videoWriter.write(outputConcatenated.astype(numpy.uint8))
    break

cv2.destroyAllWindows()

if "video_writer" in locals():
    videoWriter.release()
    print("Processing complete, video writer released ...")
