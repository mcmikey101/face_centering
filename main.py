import cv2
from ultralytics import YOLO
import dlib
import numpy as np
import imutils
from imutils import face_utils
import math


model = YOLO('./best.pt')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks (2).dat')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rects = model(frame, verbose=False)
    rects = rects[0].boxes.xyxy

    boxes = []
    for box in rects:
        boxes.append(list(map(round, box.tolist())))
    if len(boxes) != 0:
        front_box = max(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        x1, y1, x2, y2 = list(map(round, front_box))
        dlib_rect = dlib.rectangle(x1, y1, x2, y2)
        shape = predictor(frame, dlib_rect)
        shape = face_utils.shape_to_np(shape)
        r_eye = shape[36]
        l_eye = shape[45]
        dx = r_eye[0] - l_eye[0]
        dy = r_eye[1] - l_eye[1]
        alpha = math.degrees(math.atan2(dy, dx))
        box_center = [(front_box[0] + front_box[2]) / 2, (front_box[1] + front_box[3]) / 2]
        window_center = [frame.shape[1] / 2, frame.shape[0] / 2]
        x_shift, y_shift = window_center[0] - box_center[0], window_center[1] - box_center[1]
        translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]]) 
        img_translation = cv2.warpAffine(frame.copy(), translation_matrix, (frame.shape[1], frame.shape[0]))
        rotated = imutils.rotate(img_translation, 180 + alpha)
        cv2.imshow('frame', rotated)
    else: 
        cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()