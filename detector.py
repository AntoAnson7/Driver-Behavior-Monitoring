import mediapipe as mp 
import numpy as np 
import cv2 
import time
import math
import matplotlib.pyplot as plt 
import datetime
import cvlib 
from cvlib.object_detection import draw_bbox
import dlib
import torch
from imutils import face_utils
import pose_trainer

pose_classifier_model=pose_trainer.pose_classifier()




# For drowsiness detection
sleepy = 0
drowsy = 0
active = 0
status = "Active"
sleep_tot=0
drowsy_tot=0
active_tot=0

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  

# Drowsiness detection
detector=dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/DAMS/code/shape_predictor_68_face_landmarks.dat")

# Function to check drowsiness
def drowsy_check(frame):
    global sleepy, drowsy, active, status,sleep_tot,drowsy_tot,active_tot
    def compute(ptA, ptB):
        dist = np.linalg.norm(ptA - ptB)
        return dist

    def blinked(a, b, c, d, e, f):
        up = compute(b, d) + compute(c, e)
        down = compute(a, f)
        ratio = up / (2.0 * down)

        if ratio > 0.3:
            return 2
        elif (ratio > 0.25 and ratio <= 0.3):
            return 1
        else:
            return 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        if status=="Sleeping" or status=="Drowsy":
            color=(0,0,255)
        else:
            color=(0,255,0)
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38],
                             landmarks[41], landmarks[40], landmarks[39])

        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44],
                              landmarks[47], landmarks[46], landmarks[45])
        
        # print(left_blink,right_blink)

        if (left_blink == 0 or right_blink == 0):
            sleepy += 1
            drowsy = 0
            active = 0
            if (sleepy > 6):
                sleep_tot+=sleepy
                status = "Sleeping"

        if (left_blink == 1 or right_blink == 1):
            sleepy = 0
            drowsy += 1
            active = 0
            if (drowsy > 6):
                drowsy_tot+=drowsy
                status = "Drowsy"

        else:
            drowsy = 0
            sleepy = 0
            active += 1
            if (active > 6):
                active_tot+=active
                status = "Active"

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        cv2.putText(frame, f"Status : {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return frame

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


# Function for object detection
detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
def detect_obj(frame):
    toothbrush_class_index = 62
    remote_class_index = 63
    cell_phone_class_index = 67

    combined_class_index = cell_phone_class_index

    out_frame = frame.copy()

    results = detection_model(frame)

    detections = results.xyxy[0]

    combined_detections = np.concatenate((detections[detections[:, 5] == toothbrush_class_index],
                                          detections[detections[:, 5] == remote_class_index],
                                          detections[detections[:, 5] == cell_phone_class_index]))
    
    combined_detections = combined_detections[combined_detections[:, 4] >= 0.3]

    for detection in combined_detections:
        x1, y1, x2, y2, conf, class_pred = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label_text = f'{results.names[int(combined_class_index)]} {conf:.2f}'

        cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out_frame, (x1, y1 - label_height), (x1 + label_width, y1), (255, 0, 0), cv2.FILLED)
        cv2.putText(out_frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return out_frame, combined_detections[:, 4]




# Pose angle calculation function
def calculateAngle(landmark1, landmark2, landmark3):
        
    try:
        x1, y1 = (landmark1.x,landmark1.y)
        x2, y2 = (landmark2.x,landmark2.y)
        x3, y3 = (landmark3.x,landmark3.y)

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        if angle < 0:
            angle += 360
        
        return angle
    except TypeError:
        print("TYPE ERROR IN {calculateAngle}")
        return None

# Function to classify the current pose
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


        return left_elbow_angle, right_elbow_angle,left_shoulder_angle,right_shoulder_angle
    
    except KeyError:
        print("KEY ERROR IN CLASSIFYING POSE")
        return None
    
# Function to detect the pose
def detectPose(image, pose, display=True, figsize=(22, 22)):
    output_image = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    lea=0
    rea=0
    lsa=0
    rsa=0

    results = pose.process(image_rgb)
    height, width, _ = image.shape

    landmarks = []
    if results.pose_landmarks:
        lea,rea,lsa,rsa=classifyPose(results.pose_landmarks.landmark)
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
        
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
                    
    if display:
        plt.figure(figsize=figsize)
        plt.subplot(121); plt.imshow(image[:, :, ::-1]); plt.title("Original Image"); plt.axis('off');
        plt.subplot(122); plt.imshow(output_image[:, :, ::-1]); plt.title("Output Image"); plt.axis('off');
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
    else:
        return output_image,lea,rea,lsa,rsa


time1 = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    frame = cv2.flip(frame, 1)
    start = time.time()
    frame_height, frame_width, _ =  frame.shape

    # Drowsiness detection
    drowsiness_detection_frame = drowsy_check(frame.copy())

    # Object detection
    object_detection_frame,det=detect_obj(frame)

    # Pose detection
    pose_detection_frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    pose_detection_frame,lea,rea,lsa,rsa = detectPose(pose_detection_frame, pose_video, display=False)
    time2 = time.time()

    if (time2 - time1) > 0:
        frames_per_second = 1.0 / (time2 - time1)
        cv2.putText(pose_detection_frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 2)

    time1 = time2

    cv2.putText(pose_detection_frame,f"LEA : {round(360-rea,4)}" , (10, 50),cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 2)
    cv2.putText(pose_detection_frame,f"LSA : {round(rsa,4)}" , (10, 70),cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 2)
    cv2.putText(pose_detection_frame,f"REA : {round(lea,4)}" , (10, 110),cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
    cv2.putText(pose_detection_frame,f"RSA : {round(lsa,4)}" , (10, 130),cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
    
    # Face orientation
    face_orientation_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    face_orientation_results = face_mesh.process(face_orientation_frame)
    height, width, color = face_orientation_frame.shape
    face_3d = []
    face_2d = []

    face_orientation_frame = cv2.cvtColor(face_orientation_frame, cv2.COLOR_BGR2RGB)

    if face_orientation_results.multi_face_landmarks:
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
            
            cv2.putText(face_orientation_frame, f"x: {round(face_x,5)}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(face_orientation_frame, f"y: {round(face_y,5)}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(face_orientation_frame, f"z: {round(face_z,5)}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # print(x,y,z)
            face_out_text = ""
            if face_y < -14:
                face_out_text = "Looking Left"
            elif face_y > 14:
                face_out_text = "Looking Right"
            elif face_x < -14:
                face_out_text = "Looking Down"
            elif face_x > 14:
                face_out_text = "Looking Up"
            else:
                face_out_text = "Looking Forward"

            end = time.time()
            total_time = end - start
            fps = 1 / total_time
                
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
                                                                dist_matrix)
            
            p1 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
            p2 = (int(nose_3d_projection[0][0][0] + face_y * 10), int(nose_3d_projection[0][0][1] - face_x * 10))

            cv2.line(face_orientation_frame, p1, p2, (255, 0, 0), 3)

            cv2.putText(face_orientation_frame, face_out_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.putText(face_orientation_frame, f"FPS: {int(fps)}", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            mp_drawing.draw_landmarks(
                image=face_orientation_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )


    target_height = 480  
    target_width = 640  
    frame1_resized = cv2.resize(frame, (target_width, target_height))
    frame2_resized = cv2.resize(face_orientation_frame, (target_width, target_height))
    frame3_resized = cv2.resize(pose_detection_frame, (target_width, target_height))
    frame4_resized = cv2.resize(drowsiness_detection_frame, (target_width, target_height))

    top_row = np.hstack((frame1_resized, frame2_resized))
    bottom_row = np.hstack((frame3_resized, frame4_resized))

    combined_output_frame = np.vstack((top_row, bottom_row))

    # cv2.imshow('Face orientation', face_orientation_frame)
    # cv2.imshow('Pose Detection', pose_detection_frame)
    cv2.imshow('Object detection', object_detection_frame) 
    # cv2.imshow('Raw output',frame)

    # cv2.imshow('Combined output', combined_output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
