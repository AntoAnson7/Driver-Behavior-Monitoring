    plt.figure(figsize=(22,22))
    plt.subplot(121); plt.imshow(frame[:, :, ::-1]); plt.title("Original Image"); plt.axis('off');
    plt.subplot(122); plt.imshow(output_image[:, :, ::-1]); plt.title("Output Image"); plt.axis('off');
    mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
