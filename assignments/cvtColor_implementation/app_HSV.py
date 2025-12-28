import cv2
from data_path import DATA_PATH
import numpy as np


def convertToHSV(image):
    B, G, R = np.float32(cv2.split(image))
    B /= 255
    G /= 255
    R /= 255

    temp = np.maximum(B, G)
    V = np.maximum(temp, R)

    temp = np.minimum(B, G)
    delta = V - np.minimum(temp, R)

    S = np.zeros_like(V)
    S[V != 0] = (delta[V != 0] / V[V != 0]) * 255
    S = np.round(S).astype(int)

    H = np.zeros_like(V)
    H[delta != 0] = np.where(
        V[delta != 0] == R[delta != 0],
        60 * ((G[delta != 0] - B[delta != 0]) / delta[delta != 0]),
        np.where(
            V[delta != 0] == G[delta != 0],
            120 + (60 * ((B[delta != 0] - R[delta != 0]) / delta[delta != 0])),
            np.where(
                V[delta != 0] == B[delta != 0],
                240 + (60 * ((R[delta != 0] - G[delta != 0]) / delta[delta != 0])),
                0,
            ),
        ),
    )
    H[H < 0] = H[H < 0] + 360
    H = np.round(H / 2).astype(int)
    V = np.round(V * 255).astype(int)

    return np.uint8(cv2.merge([H, S, V]))


img = cv2.imread(DATA_PATH + "sample.jpg")
imgHSV = convertToHSV(img)
imgHSV_cv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

H1, S1, V1 = cv2.split(imgHSV)
H2, S2, V2 = cv2.split(imgHSV_cv)

cv2.imshow("User Defined", imgHSV)
cv2.imshow("OpenCV", imgHSV_cv)
cv2.imshow("Diff", np.abs(imgHSV - imgHSV_cv))
cv2.imshow("Diff - H", np.abs(H1 - H2))
cv2.imshow("Diff - S", np.abs(S1 - S2))
cv2.imshow("Diff - V", np.abs(V1 - V2))

cv2.waitKey(0)
cv2.destroyAllWindows()
