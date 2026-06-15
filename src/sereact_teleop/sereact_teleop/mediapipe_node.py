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

        self.publisher = self.create_publisher(Float32MultiArray, '/hand_state', 10)

        # open the first available camera and cap resolution to keep things fast
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # mediapipe hand landmarker setup
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

        # filtered_data layout: [lx, ly, l_pinch, l_angle, rx, ry, r_pinch, r_angle, l_active, r_active]
        self.filtered_data = [0.0] * 10
        self.alpha = 0.5  # smoothing factor — higher = more responsive but jerkier
        self._lock = threading.Lock()
        self.collision_active = False
        self._frame = None
        self.create_subscription(Bool, '/collision_state', self._collision_callback, 10)

        # mediapipe is too slow to run on the main thread without blocking ROS,
        # so we run camera capture and detection in a background thread
        self._running = True
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

        # the timer just reads the latest filtered data and publishes it
        # cv2.imshow also has to be called from the main thread on Linux
        self.create_timer(0.03, self.timer_callback)

    def _camera_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # mediapipe needs RGB but opencv gives BGR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(time.time() * 1000)
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            data = [0.0] * 10

            if results.hand_landmarks:
                for i, hand_landmarks in enumerate(results.hand_landmarks):
                    label = results.handedness[i][0].category_name
                    # mediapipe labels from the camera's point of view, so left and right are flipped
                    is_left = (label == 'Right')

                    wrist = hand_landmarks[0]
                    thumb_tip = hand_landmarks[4]
                    index_tip = hand_landmarks[8]

                    # pinch distance between thumb tip and index tip
                    pinch = np.sqrt((thumb_tip.x - index_tip.x)**2 +
                                    (thumb_tip.y - index_tip.y)**2)

                    # wrist rotation angle based on thumb-pinky axis relative to wrist
                    thumb_cmc = hand_landmarks[1]
                    pinky_mcp = hand_landmarks[17]
                    thumb_rel_x = thumb_cmc.x - wrist.x
                    thumb_rel_y = thumb_cmc.y - wrist.y
                    pinky_rel_x = pinky_mcp.x - wrist.x
                    pinky_rel_y = pinky_mcp.y - wrist.y
                    angle = np.arctan2(pinky_rel_y - thumb_rel_y, pinky_rel_x - thumb_rel_x)

                    # dead-man switch using 3D world landmarks — rotation-invariant
                    # middle finger tip (12) to middle finger MCP (9) distance > 5cm = finger extended
                    w = results.hand_world_landmarks[i]
                    dx = w[12].x - w[9].x
                    dy = w[12].y - w[9].y
                    dz = w[12].z - w[9].z
                    active = 1.0 if np.sqrt(dx*dx + dy*dy + dz*dz) > 0.06 else 0.0

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
                        # angle channels get a bit more smoothing since they're noisier
                        if i in [3, 7]:
                            self.filtered_data[i] = 0.6 * data[i] + 0.4 * self.filtered_data[i]
                        else:
                            self.filtered_data[i] = self.alpha * data[i] + (1 - self.alpha) * self.filtered_data[i]
                    # active flags are not smoothed — we want them to react instantly
                    self.filtered_data[8] = data[8]
                    self.filtered_data[9] = data[9]

            # draw landmarks and status on the frame
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
                # if no hands are detected we reset the active flags immediately
                # otherwise they stay stale and the robot keeps moving after the hand leaves
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
        # grab a consistent snapshot under the lock so we don't read half-updated data
        with self._lock:
            data = list(self.filtered_data)
            frame = self._frame

        msg = Float32MultiArray()
        msg.data = data
        self.publisher.publish(msg)

        # imshow must be called from the main thread on Linux, so we do it here
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
