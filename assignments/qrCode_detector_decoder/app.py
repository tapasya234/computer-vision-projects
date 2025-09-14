import cv2
import numpy
import data_path

print(data_path.DATA_PATH)
img = cv2.imread(data_path.DATA_PATH + "IDCard-Satya.png", cv2.IMREAD_UNCHANGED)
# print(img.shape)

qrDecoder = cv2.QRCodeDetector()
opencvData, bbox, rectifiedImage = qrDecoder.detectAndDecode(img)
print(bbox.shape)
print("Retval: {}, Points1: {} 2: {}".format(opencvData, bbox[0][0], bbox[0][3]))
# cv2.rectangle(
#     img,
#     pt1=(int(bbox[0][0][0]), int(bbox[0][0][1])),
#     pt2=(int(bbox[0][2][0]), int(bbox[0][2][1])),
#     color=(255, 0, 0),
#     thickness=3,
#     lineType=cv2.LINE_AA,
# )
cv2.line(
    img,
    pt1=(int(bbox[0][0][0]), int(bbox[0][0][1])),
    pt2=(int(bbox[0][1][0]), int(bbox[0][1][1])),
    color=(255, 0, 0),
    thickness=3,
    lineType=cv2.LINE_AA,
)
cv2.line(
    img,
    pt1=(int(bbox[0][1][0]), int(bbox[0][1][1])),
    pt2=(int(bbox[0][2][0]), int(bbox[0][2][1])),
    color=(255, 0, 0),
    thickness=3,
    lineType=cv2.LINE_AA,
)
cv2.line(
    img,
    pt1=(int(bbox[0][2][0]), int(bbox[0][2][1])),
    pt2=(int(bbox[0][3][0]), int(bbox[0][3][1])),
    color=(255, 0, 0),
    thickness=3,
    lineType=cv2.LINE_AA,
)
cv2.line(
    img,
    pt1=(int(bbox[0][3][0]), int(bbox[0][3][1])),
    pt2=(int(bbox[0][0][0]), int(bbox[0][0][1])),
    color=(255, 0, 0),
    thickness=3,
    lineType=cv2.LINE_AA,
)
cv2.imshow("ID", img)

cv2.imwrite(data_path.DATA_PATH + "QRCode-Output.png", img)
output = cv2.imread(data_path.DATA_PATH + "QRCode-Output.png", cv2.IMREAD_UNCHANGED)
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
