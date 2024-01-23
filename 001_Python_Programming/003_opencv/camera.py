#!/usr/bin/env python3
from imutils import face_utils
import dlib
import cv2
import numpy as np
from scipy.spatial import distance as dist

def calculate_rotation_angle(landmarks):
    left_eye=landmarks[36:42]
    right_eye=landmarks[42:48]
    nose=landmarks[27]
    
    def angle_between_points(point1,point2,center):
        a=dist.euclidean(point1,center)
        b=dist.euclidean(point2,center)
        c=dist.euclidean(point1, point2)
        angle_rad = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    left_eye_center = np.mean(left_eye, axis=0).astype(int)
    right_eye_center = np.mean(right_eye, axis=0).astype(int)

    angle = angle_between_points(left_eye_center, right_eye_center, nose)
    return angle

def rotate_image(image, angle):
    rows, cols, _=image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image


def apply_sunglasses(face, sunglasses, height, width):
    for i in range(0, height):
        for j in range(0, width):
            if sunglasses[i, j, 3] != 0:  
                face[height+i, j, :] = sunglasses[i, j, :]


# initialize face detector and then create the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# Load sunglasses images and check for errors
sunglasses_image = cv2.imread("sunglasses.png", cv2.IMREAD_COLOR)
sunglasses_image_png = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)

if sunglasses_image is None or sunglasses_image_png is None:
    print("Error:Failed to load sunglasses image.")
    exit()


height,width, _ = sunglasses_image_png.shape
resized_sunglasses_image = cv2.resize(
                                        sunglasses_image,
                                        (int(width*0.95), int(height/5)),
                                        interpolation=cv2.INTER_AREA
                                     )
resized_sunglasses_image_png = cv2.resize(
                                            sunglasses_image_png,
                                            (int(width*0.95), int(height/5)),
                                            interpolation=cv2.INTER_AREA
                                        )

# Open Webcam
cap=cv2.VideoCapture(0)


def rotate_and_resize_image(image, angle, scale):
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image
# ... (이전 코드 부분)

while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        rotation_angle = calculate_rotation_angle(shape)

        # Resize and rotate sunglasses image
        rotated_sunglasses_image = rotate_and_resize_image(
            resized_sunglasses_image_png,
            rotation_angle,
            scale=1.0  # 리사이즈 스케일을 조정해보세요
        )

        cropped_img = image[rect.top(): rect.bottom(), rect.left(): rect.right()]

        # 선글라스의 적절한 크기와 위치를 얼굴에 맞게 조절
        start_row = int(rect.top() + rect.height() * 0.35)
        end_row = start_row + rotated_sunglasses_image.shape[0]
        start_col = int(rect.left() + rect.width() * 0.15)
        end_col = start_col + rotated_sunglasses_image.shape[1]

        alpha_channel = rotated_sunglasses_image[:, :, 3] / 255.0
        alpha_channel_3d = alpha_channel[:, :, np.newaxis]

        # 얼굴 이미지와 선글라스 이미지를 블렌딩
        blended_part = (
            alpha_channel_3d * rotated_sunglasses_image[:, :, :3] +
            (1 - alpha_channel_3d) * cropped_img[
                :rotated_sunglasses_image.shape[0], :rotated_sunglasses_image.shape[1], :3
            ]
        )

        # 결과를 원본 이미지에 복사
        image[
            start_row:end_row, start_col:end_col, :3
        ] = blended_part.astype(np.uint8)

    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()


