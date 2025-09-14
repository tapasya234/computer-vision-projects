import cv2
import numpy
from data_path import DATA_PATH
import time

# Image inpainting is a class of algorithms in computer vision where the objective is to fill regions inside an image
# or a video in a way that it fits the context of its surroundings. The region is identified using a binary mask,
# and the filling is usually done by propagating information from the boundary of the region that needs to be filled.
# A common application of image inpainting is the restoration of old scanned photos.
# It is also used for removing small unwanted objects in an image.


class Sketcher:

    def __init__(self, windowName, dests, coloursFunc):
        self.prevPoint = None
        self.windowName = windowName
        self.dests = dests
        self.coloursFunc = coloursFunc
        self.dirty = False
        self.show()
        cv2.setMouseCallback(self.windowName, self.onMouse)

    def show(self):
        cv2.imshow(self.windowName, self.dests[0])
        cv2.imshow(self.windowName + ": mask", self.dests[1])

    def onMouse(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.prevPoint = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.prevPoint = None

        if self.prevPoint and flags & cv2.EVENT_FLAG_LBUTTON:
            for dest, colour in zip(self.dests, self.coloursFunc()):
                cv2.line(dest, self.prevPoint, (x, y), colour, 15)
            self.dirty = True
            self.prevPoint = (x, y)
            self.show()


img = cv2.imread(DATA_PATH + "Scan 45.jpeg", cv2.IMREAD_COLOR)

if img is None:
    print("Failed to load image")

imgMask = img.copy()
inpaintMask = numpy.zeros(img.shape[:2], numpy.uint8)

sketch = Sketcher("Image", [imgMask, inpaintMask], lambda: ((0, 255, 0), 255))

while True:
    key = cv2.waitKey()
    if key == 27:
        break
    if key == ord("t"):
        t1_telea = time.time()
        res = cv2.inpaint(
            src=imgMask,
            inpaintMask=inpaintMask,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA,
        )
        t2_telea = time.time()
        time_telea = t2_telea - t1_telea
        cv2.imshow("Telea Output", res)
    if key == ord("n"):
        t1_ns = time.time()
        res = cv2.inpaint(
            src=imgMask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv2.INPAINT_NS
        )
        t2_ns = time.time()
        time_ns = t2_ns - t1_ns
        cv2.imshow("NS Output", res)
    if key == ord("r"):
        imgMask[:] = img
        inpaintMask[:] = 0
        sketch.show()

cv2.destroyAllWindows()

print("Total Time for Telea: {}\nTotal Time for NS: {}".format(time_telea, time_ns))
