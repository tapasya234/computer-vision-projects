import cv2

# Initialize HOG parameters

# winSize - This parameter is set to the size of the window over which the descriptor is calculated.
# In classification problems, we set it to the size of the image. E.g. we set it to 64x128 for pedestrian detection.
winSize = (96, 32)

# blockSize - The size of the blue box in the image. The notion of blocks exist to tackle illumination variation.
# A large block size makes local changes less significant while a smaller block size weights local changes more.
# Typically blockSize is set to 2 x cellSize.
blockSize = (8, 8)

# blockStride - The blockStride determines the overlap between neighboring blocks
# and controls the degree of contrast normalization. Typically a blockStride is set to 50% of blockSize.
# blockStride = (8, 8)
blockStride = (4, 4)

# cellSize - The cellSize is the size of the green squares.
# It is chosen based on the scale of the features important to do the classification.
# A very small cellSize would blow up the size of the feature vector and a very large one may not capture relevant information.
cellSize = (4, 4)

# nbins - Sets the number of bins in the histogram of gradients.
# The authors of the HOG paper had recommended a value of 9 to capture gradients between 0 and 180 degrees in 20 degrees increments.
nBins = 9

# derivAperture - Size of the Sobel kernel used for derivative calculation.
derivAperture = 0

# winSigma - According to the HOG paper, it is useful to “downweight pixels near the
# edges of the block by applying a Gaussian spatial window to each pixel before accumulating orientation votes into cells”.
# winSigma is the standard deviation of this Gaussian.
# In practice, it is best to leave this parameter to default ( -1 ).
# On doing so, winSigma is automatically calculated as shown below: winSigma = ( blockSize.width + blockSize.height ) / 8
# windowSigma = 4
windowSigma = 2

# histogramNormType - In the HOG paper, the authors use four different kinds of normalization.
# OpenCV 3.2 implements only one of those types L2Hys. So, we simply use the default.
# L2Hys is simply L2 normalization followed by a threshold (L2HysThreshold) where all values above a threshold are clipped to that value.
histogramNormType = 1

# L2HysThreshold - Threshold used in L2Hys normalization.
# E.g. If the L2 norm of a vector is [0.87, 0.43, 0.22], the L2Hys normalization with L2HysThreshold = 0.8 is [0.8, 0.43, 0.22].
l2HysThreshold = 2.0000000000000001e-01

# gammaCorrection - Boolean indicating whether or not Gamma correction should be done as a pre-processing step.
gammaCorrection = 1

# nlevels - Number of pyramid levels used during detection.
# It has no effect when the HOG descriptor is used for classification.
nLevels = 64

# signedGradient - Typically gradients can have any orientation between 0 and 360 degrees.
# These gradients are referred to as “signed” gradients as opposed to “unsigned” gradients
# that drop the sign and take values between 0 and 180 degrees.
# In the original HOG paper, unsigned gradients were used for pedestrian detection.

hog = cv2.HOGDescriptor(
    winSize,
    blockSize,
    blockStride,
    cellSize,
    nBins,
    derivAperture,
    windowSigma,
    histogramNormType,
    l2HysThreshold,
    gammaCorrection,
    nLevels,
)


# Computes the hog descriptors for a given array of images. We should specify the images in a list.
def computerHOG(hog: cv2.HOGDescriptor, data):
    hogData = []
    for img in data:
        hogData.append(hog.compute(img))

    return hogData
