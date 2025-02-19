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

# Function to calculate face angle
def face_angle(shape):
    # Define two points: left corner of the left eye and right corner of the right eye
    left_eye = shape[36:42]
    right_eye = shape[42:48]

    # Calculate the midpoint between the eyes
    left_eye_midpoint = left_eye.mean(axis=0).astype("int")
    right_eye_midpoint = right_eye.mean(axis=0).astype("int")

    # Calculate the angle between the eyes
    dX = right_eye_midpoint[0] - left_eye_midpoint[0]
    dY = right_eye_midpoint[1] - left_eye_midpoint[1]
    angle = np.degrees(np.arctan2(dY, dX))

    return angle

# capture video from the webcam using cv2
cap = cv2.VideoCapture(0)
while True:
    _, image = cap.read()
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(grey, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(grey, rect)
        shape = face_utils.shape_to_np(shape)

        angle = face_angle(shape)

        cropped_img = image[rect.top(): rect.bottom(), rect.left(): rect.right()]

        # Load sunglasses images and check for errors
        sunglasses_image = cv2.imread("sunglasses.png", cv2.IMREAD_COLOR)
        sunglasses_image_png = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)

        # Resize the sunglasses images
        height, width, _ = cropped_img.shape
        resized_sunglasses_image = cv2.resize(sunglasses_image, (int(width*0.95), int(height/5)), interpolation=cv2.INTER_AREA)
        resized_sunglasses_image_png = cv2.resize(sunglasses_image_png, (int(width*0.95), int(height/5)), interpolation=cv2.INTER_AREA)

        # Rotate sunglasses image based on face angle
        rotated_sunglasses_image = cv2.warpAffine(resized_sunglasses_image, cv2.getRotationMatrix2D((width // 2, int(height/5)), -angle, 1.0), (width, height))

        for i in range(0, int(height/5)):
            for j in range(0, int(width*0.95)):
                if resized_sunglasses_image_png[i, j, 3] == 0:
                    pass
                else:
                    cropped_img[int(height/5)+i, j, :] = rotated_sunglasses_image[i, j, :]

    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
