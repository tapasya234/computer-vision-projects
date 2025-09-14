import cv2
import data_path

topLeft = ()


def drawBoundingBoxAndSave(action, x, y, flags, userdata):
    global topLeft

    if action == cv2.EVENT_LBUTTONDOWN:
        topLeft = (x, y)
    elif action == cv2.EVENT_LBUTTONUP:
        cv2.rectangle(img, topLeft, (x, y), (255, 255, 0), 2, cv2.LINE_AA)
        croppedImg = img[topLeft[1] : y, topLeft[0] : x]
        cv2.imwrite(data_path.DATA_PATH + "cropped.jpg", croppedImg)


img = cv2.imread(data_path.DATA_PATH + "sample.jpg")
winName = "Face"
cv2.namedWindow(winName)
cv2.setMouseCallback(winName, drawBoundingBoxAndSave)

k = 0
while k != 27:
    cv2.imshow(winName, img)
    cv2.putText(
        img,
        "Choose top-left corner and drag to crop",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    k = cv2.waitKey(20) & 0xFF

cv2.destroyAllWindows()
