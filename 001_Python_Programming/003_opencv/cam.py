#!/usr/bin/env python3
from imutils import face_utils
import dlib
import cv2
import numpy as np
from scipy.spatial import distance as dist

# Author: Jeong Hyeok Lim
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
# initialize facial detector using dlib
detector = dlib.get_frontal_face_detector()
# initialize predictor of facial landmarks using dlib
predictor = dlib.shape_predictor(p)

# capture video from the webcam using cv2
cap = cv2.VideoCapture(0)
while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        cropped_img = image[rect.top(): rect.bottom(), rect.left(): rect.right()]

# Load sunglasses images and check for errors
    sunglasses_image = cv2.imread("sunglasses.png", cv2.IMREAD_COLOR)
    sunglasses_image_png = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)

# Resize the sunglasses images
    height, width, _ = cropped_img.shape
    resized_sunglasses_image = cv2.resize(sunglasses_image, (int(width*0.95), int(height/5)), interpolation=cv2.INTER_AREA)
    resized_sunglasses_image_png = cv2.resize(sunglasses_image_png, (int(width*0.95), int(height/5)), interpolation=cv2.INTER_AREA)

    for i in range(0, int(height/5)):
        for j in range(0, int(width*0.95)):
            if resized_sunglasses_image_png[i, j, 3] == 0:
                pass
            else:
                cropped_img[int(height/5)+i,j,:]=resized_sunglasses_image[i,j,:]

    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
