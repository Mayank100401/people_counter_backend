import os
from flask import Flask, jsonify
from ultralytics import YOLO

# Create the Flask app
app = Flask(__name__)

# Load your model
model = YOLO('yolov8n.pt')  # or your custom model

# Conditional Webcam Access
USE_WEBCAM = os.environ.get('RAILWAY_ENVIRONMENT') != 'production'

if USE_WEBCAM:
    import cv2
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("⚠️ Warning: Could not open webcam")
    else:
        print("✅ Webcam opened successfully")

@app.route('/')
def home():
    return jsonify({"message": "Server is running 🚀"})

@app.route('/predict')
def predict():
    if not USE_WEBCAM:
        return jsonify({"error": "Webcam not available in production!"}), 400

    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Failed to capture image from webcam"}), 500

    # Run YOLO prediction
    results = model.predict(frame)

    # For simplicity, return number of detections
    return jsonify({
        "detections": len(results)
    })

# Run Flask app (only locally, Railway will handle in production)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

