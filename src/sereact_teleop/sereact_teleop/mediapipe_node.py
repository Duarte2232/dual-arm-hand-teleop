import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import cv2
import mediapipe as mp
import numpy as np

class MediaPipeNode(Node):
    def __init__(self):
        super().__init__('mediapipe_node')
        
        # Publisher
        self.publisher = self.create_publisher(Float32MultiArray, '/hand_state', 10)
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        
        # MediaPipe setup
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="src/sereact_teleop/hand_landmarker.task"),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        
        # Timer at 30Hz
        self.create_timer(0.03, self.timer_callback)

        self.filtered_data = [0.0] * 8
        self.alpha = 0.15    
    
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Initialize data with zeros
        data = [0.0] * 8
        
        if results.hand_landmarks:
            for i, hand_landmarks in enumerate(results.hand_landmarks):
                # Get handedness — flip because camera is mirrored
                label = results.handedness[i][0].category_name
                is_left = (label == 'Right')
                
                wrist = hand_landmarks[0]
                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]
                middle_mcp = hand_landmarks[9]
                
                # Pinch distance
                pinch = np.sqrt((thumb_tip.x - index_tip.x)**2 + 
                               (thumb_tip.y - index_tip.y)**2)
                
                # Wrist rotation angle
                angle = np.arctan2(middle_mcp.y - wrist.y, 
                                  middle_mcp.x - wrist.x)
                
                if is_left:
                    data[0] = wrist.x
                    data[1] = wrist.y
                    data[2] = pinch
                    data[3] = angle
                else:
                    data[4] = wrist.x
                    data[5] = wrist.y
                    data[6] = pinch
                    data[7] = angle

        # Draw landmarks on frame for debugging
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                for landmark in hand_landmarks:
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        cv2.imshow('Hand Tracking', frame)
        cv2.waitKey(1)
        
        for i in range(8):
            self.filtered_data[i] = self.alpha * data[i] + (1 - self.alpha) * self.filtered_data[i]

        msg = Float32MultiArray()
        msg.data = self.filtered_data
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MediaPipeNode()
    rclpy.spin(node)
    rclpy.shutdown()

def destroy_node(self):
    self.cap.release()
    cv2.destroyAllWindows()
    super().destroy_node()

if __name__ == '__main__':
    main()