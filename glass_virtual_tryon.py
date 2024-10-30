import cv2
import dlib
import numpy as np
import os
import logging
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlassesTryOn:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Cấu hình kích thước chuẩn cho kính
        self.TARGET_WIDTH = 400
        self.TARGET_HEIGHT = 150
        self.MIN_WIDTH = 200
        self.MAX_WIDTH = 800

        # Load glasses với error handling
        self.glasses_images = self.load_glasses_images()
        if not self.glasses_images:
            raise RuntimeError("Không thể load được bất kỳ ảnh kính nào")

        self.current_glasses_index = 0

        # Thêm các biến điều khiển cho việc hiển thị landmarks
        self.show_landmarks = False
        self.show_face_mesh = False
        self.show_glasses = False
        self.hand_gesture_enabled = False  # Thêm biến điều khiển cho hand gesture

        # Màu sắc cho landmarks và mesh
        self.LANDMARK_COLOR = (0, 255, 0)  # Xanh lá
        self.MESH_COLOR = (255, 255, 255)  # Trắng
        self.LINE_THICKNESS = 1

        # Định nghĩa các vùng facial mesh
        self.FACE_MESH_REGIONS = {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose_bridge': list(range(27, 31)),
            'nose_bottom': list(range(31, 36)),
            'right_eye': list(range(36, 42)),
            'left_eye': list(range(42, 48)),
            'outer_lips': list(range(48, 60)),
            'inner_lips': list(range(60, 68))
        }

    def toggle_glasses_display(self, enabled):
        """Bật/tắt hiển thị kính"""
        try:
            self.show_glasses = enabled
            logger.info(f"Glasses display {'enabled' if enabled else 'disabled'}")
            return True
        except Exception as e:
            logger.error(f"Error toggling glasses display: {e}")
            return False

    def toggle_landmarks(self, enabled):
        """Bật/tắt hiển thị landmarks"""
        try:
            self.show_landmarks = enabled
            logger.info(f"Facial landmarks display {'enabled' if enabled else 'disabled'}")
            return True
        except Exception as e:
            logger.error(f"Error toggling landmarks: {e}")
            return False

    def toggle_face_mesh(self, enabled):
        """Bật/tắt hiển thị face mesh"""
        try:
            self.show_face_mesh = enabled
            logger.info(f"Face mesh display {'enabled' if enabled else 'disabled'}")
            return True
        except Exception as e:
            logger.error(f"Error toggling face mesh: {e}")
            return False

    def draw_landmarks(self, frame, landmarks):
        """Vẽ các điểm landmarks trên khuôn mặt"""
        try:
            for point in range(68):
                x = landmarks.part(point).x
                y = landmarks.part(point).y
                cv2.circle(frame, (x, y), 2, self.LANDMARK_COLOR, -1)
        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")

    def draw_face_mesh(self, frame, landmarks):
        """Vẽ face mesh với các vùng được định nghĩa"""
        try:
            def draw_region(points, closed=False):
                points_array = np.array([(landmarks.part(point).x, landmarks.part(point).y)
                                         for point in points], dtype=np.int32)
                if closed:
                    cv2.polylines(frame, [points_array], True, self.MESH_COLOR,
                                  thickness=self.LINE_THICKNESS, lineType=cv2.LINE_AA)
                else:
                    for i in range(len(points) - 1):
                        pt1 = (landmarks.part(points[i]).x, landmarks.part(points[i]).y)
                        pt2 = (landmarks.part(points[i + 1]).x, landmarks.part(points[i + 1]).y)
                        cv2.line(frame, pt1, pt2, self.MESH_COLOR,
                                 thickness=self.LINE_THICKNESS, lineType=cv2.LINE_AA)

            # Vẽ từng vùng của khuôn mặt
            draw_region(self.FACE_MESH_REGIONS['jaw'])
            draw_region(self.FACE_MESH_REGIONS['right_eyebrow'])
            draw_region(self.FACE_MESH_REGIONS['left_eyebrow'])
            draw_region(self.FACE_MESH_REGIONS['nose_bridge'])
            draw_region(self.FACE_MESH_REGIONS['nose_bottom'])
            draw_region(self.FACE_MESH_REGIONS['right_eye'], closed=True)
            draw_region(self.FACE_MESH_REGIONS['left_eye'], closed=True)
            draw_region(self.FACE_MESH_REGIONS['outer_lips'], closed=True)
            draw_region(self.FACE_MESH_REGIONS['inner_lips'], closed=True)

        except Exception as e:
            logger.error(f"Error drawing face mesh: {e}")

    def validate_and_process_image(self, image_path):
        """Kiểm tra và xử lý ảnh trước khi sử dụng"""
        try:
            # Đọc ảnh với PIL để kiểm tra format
            with Image.open(image_path) as img:
                if img.mode != 'RGBA':
                    logger.warning(f"{image_path}: Converting to RGBA mode")
                    img = img.convert('RGBA')

                original_width, original_height = img.size

                # Tính toán tỷ lệ resize
                width_ratio = self.TARGET_WIDTH / original_width
                height_ratio = self.TARGET_HEIGHT / original_height
                ratio = min(width_ratio, height_ratio)

                new_width = int(original_width * ratio)
                new_height = int(original_height * ratio)

                # Giới hạn kích thước
                if new_width < self.MIN_WIDTH:
                    ratio = self.MIN_WIDTH / original_width
                elif new_width > self.MAX_WIDTH:
                    ratio = self.MAX_WIDTH / original_width

                new_width = int(original_width * ratio)
                new_height = int(original_height * ratio)

                if new_width != original_width or new_height != original_height:
                    logger.info(
                        f"Resizing {image_path} from {original_width}x{original_height} to {new_width}x{new_height}")
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                img_array = np.array(img)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
                return img_array

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None

    def optimize_alpha_channel(self, image):
        """Tối ưu hóa alpha channel"""
        try:
            if image.shape[2] != 4:
                return image

            alpha = image[:, :, 3]
            _, alpha = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)
            alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
            image[:, :, 3] = alpha
            return image

        except Exception as e:
            logger.error(f"Error optimizing alpha channel: {e}")
            return image

    def load_glasses_images(self):
        """Load tất cả ảnh kính với kiểm tra và xử lý tự động"""
        glasses_dir = "static/glasses/male"
        valid_images = []

        if not os.path.exists(glasses_dir):
            logger.error(f"Thư mục {glasses_dir} không tồn tại")
            os.makedirs(glasses_dir, exist_ok=True)
            return []

        for i in range(1, 16):  # Load 15 glasses
            image_path = os.path.join(glasses_dir, f"glass{i}.png")

            if not os.path.exists(image_path):
                logger.warning(f"Không tìm thấy file {image_path}")
                continue

            processed_image = self.validate_and_process_image(image_path)
            if processed_image is None:
                continue

            processed_image = self.optimize_alpha_channel(processed_image)

            if self.verify_image_quality(processed_image):
                valid_images.append(processed_image)
                logger.info(f"Đã load và xử lý thành công {image_path}")
            else:
                logger.warning(f"Ảnh {image_path} không đạt yêu cầu chất lượng")

        return valid_images

    def verify_image_quality(self, image):
        """Kiểm tra chất lượng ảnh sau khi xử lý"""
        try:
            if image is None:
                return False

            height, width = image.shape[:2]
            if width < self.MIN_WIDTH or width > self.MAX_WIDTH:
                logger.warning(f"Kích thước không phù hợp: {width}x{height}")
                return False

            if image.shape[2] != 4:
                logger.warning("Không có alpha channel")
                return False

            alpha = image[:, :, 3]
            if np.mean(alpha) > 250:
                logger.warning("Ảnh không có vùng trong suốt")
                return False

            return True

        except Exception as e:
            logger.error(f"Error verifying image quality: {e}")
            return False

    def get_glasses_position(self, landmarks, glasses_width):
        """Tính toán vị trí đặt kính"""
        left_eye = np.mean([(landmarks.part(36).x, landmarks.part(36).y),
                            (landmarks.part(39).x, landmarks.part(39).y)], axis=0)
        right_eye = np.mean([(landmarks.part(42).x, landmarks.part(42).y),
                             (landmarks.part(45).x, landmarks.part(45).y)], axis=0)
        nose_bridge = np.mean([(landmarks.part(27).x, landmarks.part(27).y)], axis=0)

        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        eye_distance = np.linalg.norm(right_eye - left_eye)
        scale = eye_distance / glasses_width * 2
        center = nose_bridge.astype(int)

        return center, angle, scale

    def overlay_image(self, background, foreground, location):
        """Overlay kính lên frame"""
        x, y = location
        fg_h, fg_w = foreground.shape[:2]
        bg_h, bg_w = background.shape[:2]

        if x >= bg_w or y >= bg_h:
            return background

        if x < 0:
            foreground = foreground[:, -x:]
            fg_w = foreground.shape[1]
            x = 0
        if y < 0:
            foreground = foreground[-y:, :]
            fg_h = foreground.shape[0]
            y = 0

        if x + fg_w > bg_w:
            foreground = foreground[:, :bg_w - x]
            fg_w = foreground.shape[1]
        if y + fg_h > bg_h:
            foreground = foreground[:bg_h - y, :]
            fg_h = foreground.shape[0]

        bg_region = background[y:y + fg_h, x:x + fg_w]

        if foreground.shape[2] == 4:
            fg_rgb = foreground[:, :, :3]
            alpha = foreground[:, :, 3] / 255.0
            alpha_3channel = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
            bg_region = bg_region.astype(np.float32)
            fg_rgb = fg_rgb.astype(np.float32)
            blended = cv2.add(
                cv2.multiply(bg_region, (1.0 - alpha_3channel).astype(np.float32)),
                cv2.multiply(fg_rgb, alpha_3channel.astype(np.float32))
            )
            background[y:y + fg_h, x:x + fg_w] = blended.astype(np.uint8)
        else:
            background[y:y + fg_h, x:x + fg_w] = foreground

        return background

    def apply_glasses(self, frame):
        """Apply glasses với thêm facial landmarks và mesh"""
        try:
            if frame is None:
                return frame

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)

            for face in faces:
                landmarks = self.landmark_predictor(gray, face)

                # Vẽ landmarks nếu được bật
                if self.show_landmarks:
                    self.draw_landmarks(frame, landmarks)

                # Vẽ face mesh nếu được bật
                if self.show_face_mesh:
                    self.draw_face_mesh(frame, landmarks)

                # Chỉ áp dụng kính nếu tính năng được bật
                if self.show_glasses and self.current_glasses_index < len(self.glasses_images):
                    glasses = self.glasses_images[self.current_glasses_index]
                    if glasses is None:
                        continue

                    # Tính toán kích thước kính dựa trên khuôn mặt
                    face_width = landmarks.part(16).x - landmarks.part(0).x
                    resize_ratio = face_width / glasses.shape[1]

                    # Resize kính theo khuôn mặt
                    if 0.1 < resize_ratio < 3:
                        glasses = cv2.resize(
                            glasses,
                            None,
                            fx=resize_ratio,
                            fy=resize_ratio,
                            interpolation=cv2.INTER_LANCZOS4
                        )

                    center, angle, scale = self.get_glasses_position(landmarks, glasses.shape[1])

                    M = cv2.getRotationMatrix2D(
                        (glasses.shape[1] // 2, glasses.shape[0] // 2),
                        -angle,
                        scale
                    )

                    rotated_glasses = cv2.warpAffine(
                        glasses,
                        M,
                        (glasses.shape[1], glasses.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0, 0)
                    )

                    x = int(center[0] - rotated_glasses.shape[1] // 2)
                    y = int(center[1] - rotated_glasses.shape[0] // 2)

                    frame = self.overlay_image(frame, rotated_glasses, (x, y))

            return frame

        except Exception as e:
            logger.error(f"Error in apply_glasses: {e}")
            return frame

    def change_glasses(self, direction):
        """Change glasses với validation"""
        if not self.glasses_images:
            logger.error("Không có ảnh kính nào được load")
            return

        try:
            if direction == 1:
                self.current_glasses_index = (self.current_glasses_index + 1) % len(self.glasses_images)
            else:
                self.current_glasses_index = (self.current_glasses_index - 1 + len(self.glasses_images)) % len(
                    self.glasses_images)

            logger.info(f"Changed to glasses {self.current_glasses_index + 1}")

        except Exception as e:
            logger.error(f"Error changing glasses: {e}")