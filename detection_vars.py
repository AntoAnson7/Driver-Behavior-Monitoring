# Pre-requisites
#! INSTALL yolov5
#! Download and add 'shape_predictor_68_face_landmarks.dat' to the project folder
#! Service.json for firebase admin

import mediapipe as mp 
import numpy as np 
import cv2 
import time
import math
import dlib
import torch
from imutils import face_utils
import imutils
from scipy.spatial import distance
from pygame import mixer
import datetime
from datetime import timedelta
import os
import matplotlib.pyplot as plt

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

import pose_trainer

pose_classifier_model=pose_trainer.pose_classifier()


cred = credentials.Certificate("D:/DAMS/code/Service.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'dams-4d380.appspot.com'})

mixer.init()
mixer.music.load("D:/DAMS/code/drowsy_KV/music.wav")

# Pose Detection attributes
mp_pose=mp.solutions.pose
pose=mp_pose.Pose(static_image_mode=True,min_detection_confidence=0.3,model_complexity=2)
mp_drawing=mp.solutions.drawing_utils
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Face orientation
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

  
# Function to detect the pose
# Takes (frame) returns (angles={lea,rea,lsa,rsa})
def detectPose(frame, pose=pose_video):
    output_image = frame.copy()
    height, width, _ = frame.shape
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
    landmarks = []
    if results.pose_landmarks:
        lea, rea, lsa, rsa = classifyPose(results.pose_landmarks.landmark)
        angles['lea'] = round(360-rea,4)
        angles['rea'] = round(lea,4)
        angles['lsa'] = round(rsa,4)
        angles['rsa'] = round(lsa,4)

        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
        
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    else:
        angles['lea'] = None
        angles['rea'] = None
        angles['lsa'] = None
        angles['rsa'] = None

    plt.figure(figsize=(22,22))
    plt.subplot(121); plt.imshow(frame[:, :, ::-1]); plt.title("Original Image"); plt.axis('off');
    plt.subplot(122); plt.imshow(output_image[:, :, ::-1]); plt.title("Output Image"); plt.axis('off');
    mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))


    if angles['lea'] is not None:
        pred=pose_classifier_model.predict([[int(angles['lea']),int(angles['rea']),int(angles['lsa']),int(angles['rsa'])]])[0]
    else:
        pred='SAFE'
    
    (atext_width, atext_height), _ = cv2.getTextSize(pred, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(output_image, (10, 10), (20 + atext_width, 20 + atext_height), ((0,255,0) if pred=='SAFE' else (0,0,255)), -1)
    cv2.putText(output_image, pred, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ((0, 0, 255) if pred=='SAFE' else (255,255,255)), 2)

    return pred,output_image
    
# Function for object detection
# Takes (frame) returns (detection:[1,conf] or [0,0-]) 
detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
def detect_obj(frame):
    cell_phone_class_index = 67
    toothbrush_class_index = 62
    remote_class_index = 63

    out_frame = frame.copy()

    combined_class_index = cell_phone_class_index
    
    results = detection_model(frame)
    detections = results.xyxy[0]

    combined_detections = np.concatenate((detections[detections[:, 5] == toothbrush_class_index],
                                          detections[detections[:, 5] == remote_class_index],
                                          detections[detections[:, 5] == cell_phone_class_index]))
    
    # Filter detections with confidence level over 0.3
    combined_detections = combined_detections[combined_detections[:, 4] > 0.3]
    
    (atext_width, atext_height), _ = cv2.getTextSize("SAFE", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(out_frame, (10, 10), (20 + atext_width, 20 + atext_height), (0,255,0), -1)
    cv2.putText(out_frame, "SAFE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    for detection in combined_detections:
        x1, y1, x2, y2, conf, class_pred = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label_text = f'{results.names[int(combined_class_index)]} {conf:.2f}'

        cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out_frame, (x1, y1 - label_height), (x1 + label_width, y1), (255, 0, 0), cv2.FILLED)
        cv2.putText(out_frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        (text_width, text_height), _ = cv2.getTextSize("MOBILE USAGE DETECTED", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(out_frame, (10, 10), (20 + text_width, 20 + text_height), (0,0,255), -1)
        cv2.putText(out_frame, "MOBILE USAGE DETECTED", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if len(combined_detections) > 0:
        conf = combined_detections[0, 4]
        detection = (1, conf)
    else:
        detection = (0, 0)

    return detection,out_frame

# Function for face orientation detection
# Takes (frame) returns (status : [0/1,'L/R/D/U/F'])
def face_orientation(frame):
    status=[1,'F']
    face_orientation_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    face_orientation_results = face_mesh.process(face_orientation_frame)
    height, width, color = face_orientation_frame.shape
    face_3d = []
    face_2d = []

    face_orientation_frame = cv2.cvtColor(face_orientation_frame, cv2.COLOR_BGR2RGB)

    if face_orientation_results.multi_face_landmarks:
        status=[1,'F']
        for face_landmarks in face_orientation_results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    if idx==1:
                        nose_2d=(lm.x*width,lm.y*height)
                        nose_3d=(lm.x*width,lm.y*height,lm.z*3000)
                    
                    x, y = int(lm.x * width), int(lm.y * height)
                    
                    face_2d.append([x, y])
                    
                    face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1*width
        cam_matrix = np.array([[focal_length, 0, width / 2],
                               [0, focal_length, height / 2],
                               [0, 0, 1]])

        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        if len(face_3d) > 0 and len(face_2d) > 0:
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, jac = cv2.Rodrigues(rot_vec)

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            face_x = (angles[0] * 360)
            face_y = (angles[1] * 360)
            face_z = (angles[2] * 360)
            
            face_out_text=""
            if face_y < -14:
                status =  (0,'L')            #"Looking Left"
                face_out_text = "LOOKING LEFT"
            elif face_y > 14:
                status = (0,'R')             #"Looking Right"
                face_out_text = "LOOKING RIGHT"
            elif face_x < -14:
                status = (0,'D')             #"Looking Down"
                face_out_text = "LOOKING DOWN"
            elif face_x > 14:
                status = (0,'U')             #"Looking Up"
                face_out_text = "LOOKING UP"
            else:
                status = (1,'F')             #"Looking Forward"
                face_out_text = "LOOKING FORWARD"


            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
                                                                dist_matrix)
            p1 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
            p2 = (int(nose_3d_projection[0][0][0] + face_y * 10), int(nose_3d_projection[0][0][1] - face_x * 10))

            mp_drawing.draw_landmarks(
                image=face_orientation_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

            cv2.line(face_orientation_frame, p1, p2, (255, 0, 0), 3)

            (atext_width, atext_height), _ = cv2.getTextSize(face_out_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(face_orientation_frame, (10, 10), (20 + atext_width, 20 + atext_height), (0,255,0), -1)
            cv2.putText(face_orientation_frame, face_out_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
    return status,face_orientation_frame


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def calculate_score(data):
    score=10

    def drowsiness_score(data):
        if data[0]==1:
            if data[1]/2.4 > 20:
                penalty=6.5
            else:
                penalty=4
        else:
            penalty=0

        print(f"DROWSINESS_PENALTY = {penalty}")
        return penalty

    def calculate_attention_time(data):
        attentive_seconds = 0
        inattentive_seconds = 0
        inattention_counter = 0

        for frame in data:
            if frame[0] == 1:  # If the person is attentive in this frame
                inattention_counter = 0  # Reset the inattention counter
                attentive_seconds += 0.4
            else:  # If the person is inattentive in this frame
                inattention_counter += 0.4
                if inattention_counter >= 5:  # If the person has been inattentive for 5 or more consecutive frames (1 second)
                    inattentive_seconds += 0.4

        percentage_inattentive = (inattentive_seconds / (inattentive_seconds+attentive_seconds)) * 100
        print((inattentive_seconds+attentive_seconds))
        print(percentage_inattentive)
        if percentage_inattentive<20:
            penalty = percentage_inattentive * 0.05
        elif percentage_inattentive<=40 and percentage_inattentive>20:
            penalty=percentage_inattentive*0.07
        elif percentage_inattentive<=100 and percentage_inattentive>40:
            penalty=percentage_inattentive*0.08
        
        print(f"FACE_INATTENTION_PENALTY = {penalty}")
        return (penalty,percentage_inattentive)
    
    def calculate_cellScore(data):
        det_cell_frames = 0
        penalty=0
        for frame in data:
            if frame[0] == 1:  # If object is detected
                if frame[1] > 0.3:  # If confidence level is greater than 0.3
                    det_cell_frames += 1
    
        det_cell_seconds = det_cell_frames / 2.4  # Convert frames to seconds based on frame rate
    
        if det_cell_seconds >= 20:
            penalty=4
        elif det_cell_seconds<20 and det_cell_seconds>5:
            penalty=3
        else:
            penalty=0
        
        print(f"CELL_PENALTY = {penalty}")
        return penalty,det_cell_seconds
    
    def calculate_poseScore(data):
        penalty=0
        unsafe=0
        safe=0
        for frame_val in data:
            if frame_val=='UNSAFE':
                unsafe+=1
            elif frame_val=='SAFE':
                safe+=1

        total_time=(safe+unsafe)/2.5
        safe_time=safe/2.5
        unsafe_time=unsafe/2.5

        safe_perc=(safe_time/total_time)*100
        unsafe_perc=(unsafe_time/total_time)*100

        if unsafe_perc>=75:
            penalty=2.5
        elif unsafe_perc<75 and unsafe_perc>50:
            penalty=1.5
        
        print(f"POSE_PENALTY = {penalty}")
        return penalty,unsafe_perc
        
    
    inattention_penalty,inattention_percentage=calculate_attention_time(data["face"])
    drowsiness_penalty=drowsiness_score(data["drowsy"])
    cellphone_penalty,cell_time=calculate_cellScore(data["det"])
    pose_penalty,pose_unsafe=calculate_poseScore(data["pose"])

    info={
        "inattention":inattention_percentage,
        "cell_det":0 if cellphone_penalty==0 else 1,
        "cell_time":cell_time,
        "pose_unsafe_perc":pose_unsafe
    }

    new_score=score-(inattention_penalty+drowsiness_penalty+cellphone_penalty+pose_penalty)
    if new_score<0:
        new_score=0

    return new_score,info
    
# def upload_screenshot_to_firestore(frame,user_id):
def upload_screenshot_to_firestore(data):
    # Check if the screenshot file exists
    screenshot_path = "./saves/drowsy/dr.png"
    if os.path.exists(screenshot_path):
        # Get the user ID and current date and time
        user_id = data["userid"]
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # Define the name for the screenshot in Firestore
        firestore_filename = f"{user_id}_{current_datetime}.png"

        bucket = storage.bucket()
        blob = bucket.blob(f"innattention/{firestore_filename}")
        blob.upload_from_filename(screenshot_path)

        # Generate access link for the uploaded image (with a very large expiration time)
        expiration = datetime.datetime.now() + timedelta(days=365)  # Set expiration to 365 days from now
        access_link = blob.generate_signed_url(expiration=expiration)

        return access_link
    else:
        return 0


def get_all(user_data):
    access_link=0
    s_time=time.time()

    start=datetime.datetime.now().strftime('%H:%M:%S')

    # DROWSY
    thresh = 0.3
    frame_check = 10
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor("D:/DAMS/code/shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    flag = 0
    drowsy_data = (0,0)

    data={
    "drowsy":[],
    "face":[],
    "det":[],
    "pose":[],
    "distance":15,
    "avg-speed":45,
    }

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  
    timer_start = time.time()
    print(user_data["userid"])
    
    # Flag to keep track of whether screenshot has been captured
    screenshot_taken = False
    
    while(True):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        frame = cv2.flip(frame, 1)

        # DROWSY
        dframe=frame.copy()
        dframe = imutils.resize(dframe, width=450)
        gray = cv2.cvtColor(dframe, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(dframe, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(dframe, [rightEyeHull], -1, (0, 255, 0), 1)
            # print(flag)
            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    cv2.putText(dframe, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(dframe, "****************ALERT!****************", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()
                    
                    if (not screenshot_taken):
                        save_path = "./saves/drowsy"
                        os.makedirs(save_path, exist_ok=True)
                        screenshot_path = os.path.join(save_path, "dr.png")
                        cv2.imwrite(screenshot_path, frame)
                        screenshot_taken = True

                    drowsy_data=(1,flag)  

                else:
                    (atext_width, atext_height), _ = cv2.getTextSize("Drowsiness detected", cv2.FONT_HERSHEY_SIMPLEX,0.5, 2)
                    cv2.rectangle(dframe, (10, 10), (20 + atext_width, 20 + atext_height), (0,0,255), -1)
                    cv2.putText(dframe, "Drowsiness detected", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                flag = 0
                (atext_width, atext_height), _ = cv2.getTextSize("SAFE - Not drowsy", cv2.FONT_HERSHEY_SIMPLEX,0.5, 2)
                cv2.rectangle(dframe, (10, 10), (20 + atext_width, 20 + atext_height), (0,255,0), -1)
                cv2.putText(dframe, "SAFE-NOT DROWSY", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        face_ori_status,face_frame=face_orientation(frame)
        det_status,det_frame=detect_obj(frame)
        pose_status,pose_frame=detectPose(frame)

        data["face"].append(face_ori_status)
        data["det"].append(det_status)
        data["pose"].append(pose_status)

        # Function to resize frames while preserving aspect ratio
        def resize_with_aspect_ratio(frame, width, height):
            h, w = frame.shape[:2]
            aspect_ratio = w / h
            if aspect_ratio > 1:
                new_w = width
                new_h = int(width / aspect_ratio)
            else:
                new_h = height
                new_w = int(height * aspect_ratio)
            resized_frame = cv2.resize(frame, (new_w, new_h))
            return resized_frame

        # Resize frames while preserving aspect ratio
        dframe_resized = resize_with_aspect_ratio(dframe, 500, 500)
        face_frame_resized = resize_with_aspect_ratio(face_frame, 500, 500)
        det_frame_resized = resize_with_aspect_ratio(det_frame, 500, 500)
        pose_frame_resized = resize_with_aspect_ratio(pose_frame, 500, 500)

        # Create a blank canvas for combined output frame
        max_height = max(dframe_resized.shape[0], face_frame_resized.shape[0], det_frame_resized.shape[0], pose_frame_resized.shape[0])
        max_width = max(dframe_resized.shape[1], face_frame_resized.shape[1], det_frame_resized.shape[1], pose_frame_resized.shape[1])
        combined_output_frame = np.zeros((max_height * 2, max_width * 2, 3), dtype=np.uint8)

        # Calculate positions to place resized frames
        dframe_position = ((max_height - dframe_resized.shape[0]) // 2, (max_width - dframe_resized.shape[1]) // 2)
        face_frame_position = ((max_height - face_frame_resized.shape[0]) // 2, max_width + (max_width - face_frame_resized.shape[1]) // 2)
        det_frame_position = (max_height + (max_height - det_frame_resized.shape[0]) // 2, (max_width - det_frame_resized.shape[1]) // 2)
        pose_frame_position = (max_height + (max_height - pose_frame_resized.shape[0]) // 2, max_width + (max_width - pose_frame_resized.shape[1]) // 2)

        # Fill the canvas with frames
        combined_output_frame[dframe_position[0]:dframe_position[0]+dframe_resized.shape[0], dframe_position[1]:dframe_position[1]+dframe_resized.shape[1]] = dframe_resized
        combined_output_frame[face_frame_position[0]:face_frame_position[0]+face_frame_resized.shape[0], face_frame_position[1]:face_frame_position[1]+face_frame_resized.shape[1]] = face_frame_resized
        combined_output_frame[det_frame_position[0]:det_frame_position[0]+det_frame_resized.shape[0], det_frame_position[1]:det_frame_position[1]+det_frame_resized.shape[1]] = det_frame_resized
        combined_output_frame[pose_frame_position[0]:pose_frame_position[0]+pose_frame_resized.shape[0], pose_frame_position[1]:pose_frame_position[1]+pose_frame_resized.shape[1]] = pose_frame_resized

        # Combined output
        cv2.imshow('Combined Output Frame', combined_output_frame)

        cv2.setWindowProperty("Combined Output Frame",cv2.WND_PROP_TOPMOST,1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if time.time() - timer_start > int(user_data["time"]):
            cv2.destroyAllWindows()
            cap.release()
            break
    
    end=datetime.datetime.now().strftime('%H:%M:%S')
    e_time=time.time()
    data["drowsy"]=drowsy_data

    if data["drowsy"][0]==1:
        access_link=upload_screenshot_to_firestore(user_data)

    score,info=calculate_score(data)
    avg_speed=50

    formatted_output={
        "date":datetime.datetime.now().strftime('%d/%m/%Y'),
        "ride_duration":f"{(e_time-s_time)/60}",
        "start_time":start,
        "end_time":end,
        "drowsiness_status":drowsy_data[0],
        "score":score,
        "inattention":info["inattention"],
        "cellphone_det":info["cell_det"],
        "cell_time":info["cell_time"],
        "links":access_link if access_link!=0 else "no images",
        "avg_speed":avg_speed,
        "distance":float(f"{avg_speed*((e_time-s_time)/3600)}"),
        "pose_unsafe_perc":info["pose_unsafe_perc"]
    }

    return formatted_output




# Test line
# print(get_all({"userid":"ABCD","time":120}))