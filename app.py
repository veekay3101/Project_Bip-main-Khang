import os
import cv2
import cvzone
import math
from flask import Flask, request, render_template, jsonify, Response
from ultralytics import YOLO
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import time
import torch

app = Flask(__name__)

# Tải mô hình YOLO
model = YOLO('yolov8n.pt')
if torch.cuda.is_available():
    model.to('cuda')
classnames = model.names

# Thư mục lưu tệp tải lên và xử lý
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có tệp được tải lên'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Không có tệp được chọn'}), 400

    # Lưu video tải lên
    video_path = os.path.join(UPLOAD_FOLDER, 'input_video.mp4')
    file.save(video_path)

    # Xử lý video
    output_path = os.path.join(UPLOAD_FOLDER, 'output_video.mp4')
    fall_detected = process_video(video_path, output_path)

    return render_template('result.html', video_url='/static/uploads/output_video.mp4', fall_detected=fall_detected)

@app.route('/webcam', methods=['POST'])
def webcam():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Không có dữ liệu hình ảnh'}), 400

    # Chuyển đổi base64 thành hình ảnh
    img_data = base64.b64decode(data['image'].split(',')[1])
    img = Image.open(BytesIO(img_data)).convert('RGB')
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Xử lý khung hình
    fall_detected, annotated_frame = process_frame(frame)

    # Chuyển đổi khung hình đã xử lý thành base64
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'image': f'data:image/jpeg;base64,{img_str}',
        'fall_detected': fall_detected
    })

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    prev_time = 0
    fall_timer = 0
    fall_detected = False
    frame_count = 0
    frame_skip = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (640, 480))
        fall_detected_in_frame, annotated_frame = process_frame(frame)

        if fall_detected_in_frame:
            fall_detected = True
            if time.time() - fall_timer > 5:
                fall_timer = time.time()

        out.write(annotated_frame)

    cap.release()
    out.release()
    return fall_detected

def process_frame(frame):
    results = model(frame, verbose=False)
    any_fall = False

    for info in results:
        boxes = info.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            class_name = classnames.get(cls_id, f"Unknown({cls_id})")
            confidence = math.ceil(conf * 100)

            if class_name != 'person' or confidence < 80:
                continue

            width = x2 - x1
            height = y2 - y1
            is_fall = width > height

            color = (0, 0, 255) if is_fall else (0, 255, 0)
            cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6, colorC=color)
            cvzone.putTextRect(frame, f'{class_name} {confidence}%', [x1 + 8, y1 - 12], thickness=2, scale=1)

            if is_fall:
                any_fall = True
                cvzone.putTextRect(frame, ' PHÁT HIỆN TÉ NGÃ ', [x1, y2 + 20], scale=2, thickness=2, colorR=(0, 0, 255))

    curr_time = time.time()
    fps = 1 / (curr_time - process_frame.prev_time) if hasattr(process_frame, 'prev_time') else 0
    process_frame.prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return any_fall, frame

if __name__ == '__main__':
    app.run(debug=True)