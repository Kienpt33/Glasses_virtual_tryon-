import cv2
import mediapipe as mp
from skimage.io import imshow


class HandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.last_hand_x = None

    def detect_hand_movement(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            current_x = index_tip.x

            if self.last_hand_x is not None:
                if current_x - self.last_hand_x > 0.02:
                    self.last_hand_x = current_x
                    return 1  # Chuyển động sang phải
                elif self.last_hand_x - current_x > 0.02:
                    self.last_hand_x = current_x
                    return -1  # Chuyển động sang trái

            self.last_hand_x = current_x
        else:
            self.last_hand_x = None

        return 0  # Không có chuyển động

    def close(self):
        self.hands.close()