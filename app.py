from flask import Flask, jsonify
from ultralytics import YOLO
import cv2
import threading

app = Flask(__name__)

people_count = 0

model = YOLO('yolov8n.pt')  # use yolov8s.pt for better

def detect_people():
    global people_count
    cap = cv2.VideoCapture(0)  # Use the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        results = model.predict(source=frame, conf=0.5, classes=[0], verbose=False)
        people_count = len(results[0].boxes)

detect_thread = threading.Thread(target=detect_people)
detect_thread.start()

@app.route('/')
def home():
    return "People Counter is Running!"

@app.route('/count')
def count():
    return jsonify(count=people_count)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
