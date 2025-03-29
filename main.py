import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

model = YOLO('face_centering/best.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    res = model(frame, verbose=False)

    boxes = []
    for box in res[0].boxes:
        boxes.append(list(map(round, box.xyxy[0].tolist())))
    if len(boxes) != 0:
        front_box = max(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        box_center = [(front_box[0] + front_box[2]) / 2, (front_box[1] + front_box[3]) / 2]
        window_center = [frame.shape[1] / 2, frame.shape[0] / 2]
        x_shift, y_shift = window_center[0] - box_center[0], window_center[1] - box_center[1]
        translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]]) 
        img_translation = cv2.warpAffine(frame.copy(), translation_matrix, (frame.shape[1], frame.shape[0]))
        cv2.imshow('frame', img_translation)
    else: 
        cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()