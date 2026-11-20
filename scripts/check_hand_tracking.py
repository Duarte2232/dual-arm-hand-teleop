#!/usr/bin/env python3
"""Standalone webcam check of the hand tracker (no ROS needed).

Opens the first camera, runs the same MediaPipe hand landmarker model the
teleop uses, and draws the landmarks. Press q to quit.

    python3 scripts/check_hand_tracking.py
"""
import os
import time

import cv2
import mediapipe as mp

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..',
    'src', 'dual_arm_hand_teleop', 'config', 'hand_landmarker.task')

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV gives BGR, MediaPipe expects RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

    h, w, _ = frame.shape
    for hand in results.hand_landmarks or []:
        for landmark in hand:
            cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 5, (0, 255, 0), -1)

    cv2.imshow('MediaPipe Hands', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
