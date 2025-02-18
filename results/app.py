import os
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from threading import Thread

# Инициализация Flask
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Загрузка предобученной модели (примерная заглушка)
class PotholeDetector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # YOLOv5

    def detect(self, frame):
        results = self.model(frame)  # Детекция
        labels, boxes = results.xyxy[0][:, -1].cpu().numpy(), results.xyxy[0][:, :-1].cpu().numpy()
        pothole_count = sum(1 for label in labels if label == 0)  # Допустим, класс 0 — яма

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame, pothole_count

detector = PotholeDetector()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, pothole_count = detector.detect(frame)

        _, buffer = cv2.imencode(".jpg", processed_frame)
        frame_bytes = buffer.tobytes()
        
        socketio.emit("video_frame", {"image": frame_bytes, "potholes": pothole_count})
    cap.release()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    file = request.files["video"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Запуск обработки видео в отдельном потоке
    Thread(target=process_video, args=(file_path,)).start()
    return "Видео загружено и обрабатывается."

if __name__ == "__main__":
    socketio.run(app, debug=True)
