import cv2
from data_path import DATA_PATH


def pencilSketchFilter(img, arguments=0):
    imgGrayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.adaptiveThreshold(
        imgGrayscale,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        7,
    )
    return cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR)


def cartoonFilter(image, arguments=0):
    imgGrayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.adaptiveThreshold(
        imgGrayscale,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        19,
        5,
    )
    imgThresh = cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR)
    imgBlur = cv2.bilateralFilter(image, 9, 30, 25)
    return cv2.bitwise_and(imgBlur, imgThresh)


def cartoonify(image, arguments=0):
    ### YOUR CODE HERE
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(img_gray, 3)
    edges = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )

    color = cv2.bilateralFilter(image, 30, 80, 80)
    contours = pencilSketch(image)
    return cv2.bitwise_and(color, color, mask=contours[:, :, 0])


# In[34]:


def pencilSketch(image, arguments=0):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.5)
    lapl = cv2.Laplacian(blurred, cv2.CV_8U, ksize=3, scale=2, delta=1)
    _, dst = cv2.threshold(lapl, 16, 255, cv2.THRESH_BINARY)

    return cv2.cvtColor(255 - dst, cv2.COLOR_GRAY2BGR)


img = cv2.imread(DATA_PATH + "trump.jpg")
cv2.imshow("Original", img)
cv2.imshow("Pencil Sketch", pencilSketch(img))
cv2.imshow("Cartoonify", cartoonify(img))
cv2.imshow("Pencil Sketch Filter", pencilSketchFilter(img))
cv2.imshow("Cartoonify Filter", cartoonFilter(img))

cv2.waitKey(0)
cv2.destroyAllWindows()
