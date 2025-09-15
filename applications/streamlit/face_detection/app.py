import streamlit
import numpy
import cv2
import os
from io import BytesIO
import base64
from PIL import Image


def getFilePath(fileName: str):
    return os.path.join(
        os.getcwd(),
        "python3/computer-vision-projects/applications/streamlit/face_detection/"
        + fileName,
    )


@streamlit.cache_resource
def loadModel():
    modelFile = getFilePath("res10_300x300_ssd_iter_140000_fp16.caffemodel")
    configFile = getFilePath("deploy.prototxt")
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net


def detectFacesOnImg(net, img):
    blob = cv2.dnn.blobFromImage(
        image=img,
        scalefactor=1.0,
        size=(300, 300),
        mean=[104, 117, 123],
        swapRB=False,
        crop=False,
    )

    net.setInput(blob)
    # Get Detections.
    return net.forward()


# Function for annotating the image with bounding boxes for each detected face.
def processDetection(img, detections, confThreshold=0.5):
    bboxes = []
    frameHeight = img.shape[0]
    frameWidth = img.shape[1]

    boxThickness = max(1, int(round(frameHeight / 200)))
    # Loop over all detections and draw bounding boxed around each face.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confThreshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(
                img, (x1, y1), (x2, y2), (0, 255, 0), boxThickness, cv2.LINE_8
            )
    return img, bboxes


# Function to generate a download link for output file.
def getImgDownloadLink(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


streamlit.title("FACE DETECTION (OpenCV Deep Learning)")
streamlit.text(
    "This app will require the user to upload a image and will try to detect faces on that image. It also lets the user customize ..."
)

rawImg = streamlit.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

net = loadModel()
if rawImg is not None:
    # Read the file and convert it into an OpenCV Image
    rawBytes = numpy.asarray(bytearray(rawImg.read()), dtype=numpy.uint8)
    inputImg = cv2.imdecode(rawBytes, flags=cv2.IMREAD_COLOR)

    placeholders = streamlit.columns(2)
    placeholders[0].image(inputImg, channels="BGR", caption="Input Image")
    # placeholders[0].text("Input Image")

    confidenceThreshold = streamlit.slider(
        "Set Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

    # Call the face detection model to detect faces in the image.
    detections = detectFacesOnImg(net, inputImg)

    # Process the detections based on the current confidence threshold.
    outputImg, _ = processDetection(
        inputImg, detections, confThreshold=confidenceThreshold
    )

    placeholders[1].image(outputImg, channels="BGR", caption="Output Image")

    # Create a link for downloading the output file.
    outputImg = Image.fromarray(outputImg[:, :, ::-1])
    streamlit.markdown(
        getImgDownloadLink(outputImg, "face_output.jpg", "Download Output Image"),
        unsafe_allow_html=True,
    )
