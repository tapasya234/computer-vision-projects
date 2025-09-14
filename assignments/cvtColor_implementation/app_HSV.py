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
    s = S
    S = np.round(S).astype(int)
    # S = np.where(np.round(S, 1) - np.floor(S) >= 0.5, np.ceil(S), np.round(S))

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
    h = H / 2
    H = np.round(H / 2).astype(int)
    V = np.round(V * 255).astype(int)

    imgHSV_cv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H2, S2, V2 = cv2.split(imgHSV_cv)

    countS = 0
    countH = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if S[i][j] != S2[i][j]:
                print(
                    "i: {} j: {} S_rounded: {} S_openCV: {} S_float: {}".format(
                        i, j, S[i][j], S2[i][j], s[i][j]
                    )
                )
                countS += 1
            if H[i][j] != H2[i][j]:
                print(
                    "i: {} j: {} H_rounded: {} H_openCV: {} H_float: {}".format(
                        i, j, H[i][j], H2[i][j], h[i][j]
                    )
                )
                countH += 1

    print(countS)
    print(countH)
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
