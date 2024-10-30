from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
from glass_virtual_tryon import GlassesTryOn
from Dectect_hand import HandGestureDetector
from deepface import DeepFace
import threading
import time
import mediapipe as mp

app = Flask(__name__)

# Khởi tạo các class và biến global
glasses_try_on = GlassesTryOn()
hand_detector = HandGestureDetector()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Biến điều khiển cho face recognition
face_recognition_active = False
current_result = {"age": None, "gender": None}
result_lock = threading.Lock()
analysis_in_progress = False

def predict_gender_age(face_img):
    global current_result, analysis_in_progress
    analysis_in_progress = True
    try:
        result = DeepFace.analyze(
            face_img, actions=["age", "gender"], enforce_detection=False
        )
        with result_lock:
            current_result["age"] = result[0]["age"]
            current_result["gender"] = result[0]["dominant_gender"]
    except Exception as e:
        print(f"Lỗi khi phân tích khuôn mặt: {str(e)}")
    finally:
        analysis_in_progress = False

def analyze_face_thread(face_img):
    thread = threading.Thread(target=predict_gender_age, args=(face_img,))
    thread.start()

def draw_result(frame, x, y, w, h, age, gender):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    text = f"{gender}, {age}" if age is not None and gender is not None else "Analyzing..."
    cv2.putText(
        frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2
    )

def gen_frames():
    global face_recognition_active
    cap = cv2.VideoCapture(0)
    last_prediction_time = time.time()
    prediction_interval = 5  # Predict mỗi 5 giây
    last_glasses_change_time = 0
    glasses_change_cooldown = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lật ngược video theo chiều ngang
        frame = cv2.flip(frame, 1)

        current_time = time.time()

        if face_recognition_active:
            # Xử lý nhận diện khuôn mặt và đoán tuổi
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_img = frame[y:y + h, x:x + w]

                with result_lock:
                    age = current_result["age"]
                    gender = current_result["gender"]

                if age is None or gender is None:
                    if not analysis_in_progress:
                        analyze_face_thread(face_img.copy())
                elif current_time - last_prediction_time >= prediction_interval:
                    if not analysis_in_progress:
                        last_prediction_time = current_time
                        analyze_face_thread(face_img.copy())

                draw_result(frame, x, y, w, h, age, gender)
        else:
            # Xử lý virtual try-on glasses
            frame = glasses_try_on.apply_glasses(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/change_glasses/<int:glasses_id>', methods=['POST'])
def change_glasses(glasses_id):
    global face_recognition_active
    face_recognition_active = False  # Tắt face recognition khi bắt đầu thử kính
    glasses_try_on.change_glasses(glasses_id)
    return {'status': 'success'}

@app.route('/toggle_feature/<feature>', methods=['POST'])
def toggle_feature(feature):
    global face_recognition_active
    try:
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Request phải ở định dạng JSON'
            }), 400

        data = request.get_json()
        if 'enabled' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Thiếu tham số "enabled"'
            }), 400

        enabled = data['enabled']
        face_recognition_active = False  # Tắt face recognition khi bật bất kỳ tính năng nào

        # Xử lý các tính năng
        success = False
        message = ""

        if feature == 'face-tracking':
            success = glasses_try_on.toggle_glasses_display(enabled)
            message = "Hiển thị kính"
        elif feature == 'landmarks':
            success = glasses_try_on.toggle_landmarks(enabled)
            message = "Hiển thị landmarks"
        elif feature == 'face-mesh':
            success = glasses_try_on.toggle_face_mesh(enabled)
            message = "Hiển thị face mesh"
        elif feature == 'hand-gesture':
            success = glasses_try_on.toggle_hand_gesture(enabled)
            message = "Điều khiển cử chỉ tay"
        else:
            return jsonify({
                'status': 'error',
                'message': 'Tính năng không hợp lệ'
            }), 400

        if success:
            return jsonify({
                'status': 'success',
                'message': f'Đã {"bật" if enabled else "tắt"} {message.lower()}'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Không thể {("bật" if enabled else "tắt")} {message.lower()}'
            }), 500

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/reset_face_recognition', methods=['POST'])
def reset_face_recognition():
    """API endpoint để reset về chế độ nhận diện khuôn mặt"""
    global face_recognition_active, current_result
    face_recognition_active = True
    with result_lock:
        current_result = {"age": None, "gender": None}
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)