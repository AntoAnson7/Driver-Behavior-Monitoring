import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.models import save_model
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)

def pose_classifier():
    safe_data = pd.read_excel("datasets/safe.xlsx")
    unsafe_data = pd.read_excel("datasets/unsafe.xlsx")

    safe_data['label'] = 'SAFE'
    unsafe_data['label'] = 'UNSAFE'

    data = pd.concat([safe_data, unsafe_data], ignore_index=True)
    X = data[['lea', 'rea', 'lsa', 'rsa']]
    y = data['label']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Classifier Accuracy: {test_accuracy:.2f}")

    return model


# #! ____________________________________________________________________________________________________________________________________________________________________________________-
# #! ____________________________________________________________________________________________________________________________________________________________________________________-
# import mediapipe as mp 
# import numpy as np 
# import cv2 
# import time
# import math
# import dlib
# import torch
# from imutils import face_utils
# import imutils
# from scipy.spatial import distance
# from pygame import mixer
# import datetime
# from datetime import timedelta
# import os
# # Pose Detection attributes
# mp_pose=mp.solutions.pose
# pose=mp_pose.Pose(static_image_mode=True,min_detection_confidence=0.3,model_complexity=2)
# mp_drawing=mp.solutions.drawing_utils
# pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)


# def calculateAngle(landmark1, landmark2, landmark3):
#     try:
#         x1, y1 = (landmark1.x, landmark1.y)
#         x2, y2 = (landmark2.x, landmark2.y)
#         x3, y3 = (landmark3.x, landmark3.y)

#         angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

#         if angle < 0:
#             angle += 360

#         return angle
#     except TypeError:
#         print("TYPE ERROR IN {calculateAngle}")
#         return None

# def classifyPose(landmarks):
#     try:
#         left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
#                                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
#                                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

#         right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
#                                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
#                                             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

#         left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
#                                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
#                                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

#         right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
#                                                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
#                                                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

#         return [[left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle]]

#     except KeyError:
#         print("KEY ERROR IN CLASSIFYING POSE")
#         return None
    
# def get_all():
#     cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  

#     while(True):
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture frame")
#             break
#         frame = cv2.flip(frame, 1)

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#         results = pose.process(frame_rgb)
#         if results.pose_landmarks:
#             angles = classifyPose(results.pose_landmarks.landmark)

#         else:
#             angles=[[0,0,0,0]]

#         value=pose_classifier_model.predict(angles)
        
#         # Put the text on the image
#         cv2.putText(frame, value[0], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#         cv2.imshow('frame',frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             break

# get_all()