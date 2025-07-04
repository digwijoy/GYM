from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np

print("✅ Flask app is starting...")

from process_frame_curling import process_frame_curling
from process_frame_squats import process_frame_squats
from process_frame_lunges import process_frame_lunges

app = Flask(__name__)

# ✅ Mediapipe pose detector
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/exercise/<exercise_name>')
def exercise_page(exercise_name):
    return render_template(f"{exercise_name}_trainer.html")

@app.route('/video_feed/<exercise>')
def video_feed(exercise):
    return Response(
        generate_frames(exercise),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/test_camera')
def test_camera():
    cap = cv2.VideoCapture(0)
    result = cap.isOpened()
    cap.release()
    return jsonify({'camera_available': result})

@app.route('/release_camera')
def release_camera():
    return jsonify(success=True)

def generate_frames(exercise):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("🎥 Trying to open webcam...")
    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam.")
        blank = 255 * np.ones((480, 640, 3), dtype=np.uint8)
        ret, buffer = cv2.imencode('.jpg', blank)
        frame_bytes = buffer.tobytes()
        for _ in range(100):
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return

    print(f"✅ Webcam stream started for exercise: {exercise}")
    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Failed to read frame.")
            break

        try:
            # Process based on exercise
            if exercise == "bicep_curl":
                frame, status = process_frame_curling(frame, pose)
            elif exercise == "squats":
                frame, status = process_frame_squats(frame, pose)
            elif exercise == "lunges":
                frame, status = process_frame_lunges(frame, pose)
            else:
                print(f"❓ Unknown exercise: {exercise}")
                status = "Unknown exercise"

            if status:
                cv2.putText(
                    frame,
                    status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("❌ Failed to encode frame.")
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"🔥 ERROR during frame processing: {e}")
            break

    cap.release()
    print("📴 Webcam released.")

if __name__ == '__main__':
    print("🚀 Running on http://127.0.0.1:5000")
    app.run(debug=True)
