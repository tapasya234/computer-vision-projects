import cv2
from data_path import DATA_PATH
import numpy as np


def rbgToHsv(r, g, b):
    # print("R: {} G: {} B: {}".format(r, g, b))
    r = r / 255
    g = g / 255
    b = b / 255

    maxRGB = max(r, g, b)
    minRGB = min(r, g, b)

    s = 0
    h = 0
    if maxRGB != 0:
        delta = maxRGB - minRGB
        s = (delta / maxRGB) * 255
        print("s: {} round: {} floor: {}".format(s, np.round(s, 1), np.floor(s)))
        if np.round(s, 1) - np.floor(s) >= 0.5:
            s = np.ceil(s)
        else:
            s = np.round(s)

        print("S org: {} rounded: {}".format((delta / maxRGB) * 255, s))

        if maxRGB == r:
            h = 60 * ((g - b) / delta) + 360
        elif maxRGB == g:
            h = 60 * ((b - r) / delta) + 120
        else:
            h = 60 * ((r - g) / delta) + 240

        if h < 0:
            h += 360

    print("H: {} S: {} V: {}".format(h / 2, s, maxRGB * 255))
    return np.round(h / 2), np.round(s), np.round(maxRGB * 255)


def convertToHsv(image):
    b, g, r = cv2.split(image)

    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    # for
    print("User designed HSV: ", rbgToHsv(r[511][352], g[511][352], b[511][352]))
    print("CV HSV: ", h[511][352], s[511][352], v[511][352])


img = cv2.imread(DATA_PATH + "sample.jpg")
convertToHsv(img)
