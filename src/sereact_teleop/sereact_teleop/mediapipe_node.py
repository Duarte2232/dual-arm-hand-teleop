#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import threading
from ament_index_python.packages import get_package_share_directory

class MediaPipeNode(Node):
    def __init__(self):
        super().__init__('mediapipe_node')

        # Publisher
        self.publisher = self.create_publisher(Float32MultiArray, '/hand_state', 10)

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # MediaPipe setup
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        model_path = os.path.join(
            get_package_share_directory('sereact_teleop'),
            'config', 'hand_landmarker.task'
        )
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        self.filtered_data = [0.0] * 10  # 8 hand values + 2 active flags
        self.alpha = 0.5
        self._lock = threading.Lock()
        self.collision_active = False
        self._frame = None
        self.create_subscription(Bool, '/collision_state', self._collision_callback, 10)

        # Run camera + MediaPipe in background thread
        self._running = True
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

        # Timer only publishes — no heavy work
        self.create_timer(0.03, self.timer_callback)

    def _camera_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(time.time() * 1000)
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            data = [0.0] * 10  # 8 hand values + 2 active flags

            if results.hand_landmarks:
                for i, hand_landmarks in enumerate(results.hand_landmarks):
                    label = results.handedness[i][0].category_name
                    is_left = (label == 'Right')

                    wrist = hand_landmarks[0]
                    thumb_tip = hand_landmarks[4]
                    index_tip = hand_landmarks[8]

                    pinch = np.sqrt((thumb_tip.x - index_tip.x)**2 +
                                    (thumb_tip.y - index_tip.y)**2)

                    thumb_cmc = hand_landmarks[1]
                    pinky_mcp = hand_landmarks[17]
                    thumb_rel_x = thumb_cmc.x - wrist.x
                    thumb_rel_y = thumb_cmc.y - wrist.y
                    pinky_rel_x = pinky_mcp.x - wrist.x
                    pinky_rel_y = pinky_mcp.y - wrist.y
                    angle = np.arctan2(pinky_rel_y - thumb_rel_y, pinky_rel_x - thumb_rel_x)

                    # Dead-man switch: use 3D world landmarks so rotation doesn't affect detection
                    w = results.hand_world_landmarks[i]
                    dx = w[12].x - w[9].x
                    dy = w[12].y - w[9].y
                    dz = w[12].z - w[9].z
                    active = 1.0 if np.sqrt(dx*dx + dy*dy + dz*dz) > 0.05 else 0.0

                    if is_left:
                        data[0] = wrist.x
                        data[1] = wrist.y
                        data[2] = pinch
                        data[3] = angle
                        data[8] = active
                    else:
                        data[4] = wrist.x
                        data[5] = wrist.y
                        data[6] = pinch
                        data[7] = angle
                        data[9] = active

                with self._lock:
                    for i in range(8):
                        if i in [3, 7]:
                            self.filtered_data[i] = 0.6 * data[i] + 0.4 * self.filtered_data[i]
                        else:
                            self.filtered_data[i] = self.alpha * data[i] + (1 - self.alpha) * self.filtered_data[i]
                    # Active flags are not smoothed — respond instantly
                    self.filtered_data[8] = data[8]
                    self.filtered_data[9] = data[9]

            h, w, _ = frame.shape
            for i, hand_landmarks in enumerate(results.hand_landmarks):
                label = results.handedness[i][0].category_name
                is_left = (label == 'Right')
                active = data[8] if is_left else data[9]
                dot_color  = (0, 255, 0) if active > 0.5 else (0, 0, 255)
                status_text = 'ACTIVE' if active > 0.5 else 'HOLD'
                side = 'L' if is_left else 'R'

                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, dot_color, -1)

                wrist = hand_landmarks[0]
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                cv2.circle(frame, (wx, wy), 14, dot_color, -1)
                cv2.putText(frame, f'{side}: {status_text}', (wx + 18, wy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, dot_color, 2)

            if not results.hand_landmarks:
                with self._lock:
                    self.filtered_data[8] = 0.0
                    self.filtered_data[9] = 0.0
                cv2.putText(frame, 'NO HANDS DETECTED', (w // 2 - 140, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

            if self.collision_active:
                cv2.rectangle(frame, (4, 4), (w - 4, h - 4), (0, 0, 255), 4)
                cv2.putText(frame, 'COLLISION BLOCKED', (w // 2 - 140, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            with self._lock:
                self._frame = frame.copy()

    def _collision_callback(self, msg):
        self.collision_active = msg.data

    def timer_callback(self):
        with self._lock:
            data = list(self.filtered_data)
            frame = self._frame

        msg = Float32MultiArray()
        msg.data = data
        self.publisher.publish(msg)

        if frame is not None:
            cv2.imshow('Hand Tracking', frame)
            cv2.waitKey(1)


    def destroy_node(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MediaPipeNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
