# Data collection for pose classifier

import mediapipe as mp 
import os
import cv2 
import time
import math
import datetime
import pandas as pd

# Pose Detection attributes
mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Takes (frame) returns (angles={lea,rea,lsa,rsa})
def detectPose(frame, pose=pose_video):
    def calculateAngle(landmark1, landmark2, landmark3):
        try:
            x1, y1 = (landmark1.x, landmark1.y)
            x2, y2 = (landmark2.x, landmark2.y)
            x3, y3 = (landmark3.x, landmark3.y)

            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

            if angle < 0:
                angle += 360

            return angle
        except TypeError:
            print("TYPE ERROR IN {calculateAngle}")
            return None

    def classifyPose(landmarks):
        try:
            left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

            right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

            left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

            right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

            return left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle

        except KeyError:
            print("KEY ERROR IN CLASSIFYING POSE")
            return None

    angles = {}
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        lea, rea, lsa, rsa = classifyPose(results.pose_landmarks.landmark)
        angles['lea'] = round(360 - rea, 4)
        angles['rea'] = round(lea, 4)
        angles['lsa'] = round(rsa, 4)
        angles['rsa'] = round(lsa, 4)
    else:
        angles['lea'] = None
        angles['rea'] = None
        angles['lsa'] = None
        angles['rsa'] = None

    return angles
    
def actions(user_data):
    start = datetime.datetime.now().strftime('%H-%M-%S')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  
    timer_start = time.time()
    
    data = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        frame = cv2.flip(frame, 1)
        frame= cv2.rotate(frame,cv2.ROTATE_180)

        pose_status = detectPose(frame)
        data.append(pose_status)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if time.time() - timer_start > int(user_data["time"]):
            cv2.destroyAllWindows()
            cap.release()
            break
    
    df = pd.DataFrame(data)
    df['label'] = 'unsafe'
    filename = 'unsafe_{}.xlsx'.format(start)
    df.to_excel(filename, index=False)
    return filename

print(actions({"userid": "TEST", "time": 120}))
