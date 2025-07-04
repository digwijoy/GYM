from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime
import time

# ✅ Import your exercise processors
from process_frame_curling import process_frame_curling
from process_frame_squats import process_frame_squats
from process_frame_lunges import process_frame_lunges
from process_frame_tricep_kickback import process_frame_tricep_kickback

print("✅ Flask app is starting...")

app = Flask(__name__)

# ✅ MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ✅ For logging last reps
last_reps = {
    "bicep_curl": 0,
    "squats": 0,
    "lunges": 0,
    "tricep_kickback": 0
}

# ✅ Active user sessions
user_sessions = {}


# ✅ Save workout log
def save_workout(exercise, reps):
    data = {
        "exercise": exercise,
        "reps": reps,
        "timestamp": str(datetime.now())
    }
    with open("workout_log.json", "a") as f:
        f.write(json.dumps(data) + "\n")
    print(f"✅ Workout saved: {data}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start_workout', methods=['POST'])
def start_workout():
    data = request.get_json()
    username = data.get('username')
    exercise = data.get('exercise')
    reps_per_set = data.get('reps_per_set')
    sets = data.get('sets')
    rest_time = data.get('rest_time', 60)  # Default rest time 60s

    if not username or not exercise:
        return jsonify({"error": "Missing data"}), 400

    user_sessions[username] = {
        "exercise": exercise,
        "reps_per_set": int(reps_per_set),
        "sets": int(sets),
        "current_set": 1,
        "reps_done_in_set": 0,
        "workout_complete": False,
        "prev_reps_done_in_set": None,
        "prev_set": None,
        "last_message": "",
        "rest_time": int(rest_time),
        "is_resting": False,
        "rep_history": [],
        "calorie_history": [],
        "calories_burned": 0,
        "muscle_progress": 0
    }

    print(f"✅ New session for {username}: {user_sessions[username]}")
    return jsonify({"message": "Workout session started!"})


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
    print("📴 Release camera endpoint hit.")
    return jsonify(success=True)


@app.route('/reset_counters/<exercise>')
def reset_counters(exercise):
    if exercise in last_reps:
        save_workout(exercise, last_reps[exercise])
        last_reps[exercise] = 0
        return jsonify({"message": f"{exercise.capitalize()} counters reset and saved."})
    return jsonify({"error": "Invalid exercise name."}), 400


@app.route('/workout_status/<username>')
def workout_status(username):
    session = user_sessions.get(username)
    if session:
        return jsonify({
            "exercise": session["exercise"],
            "current_set": session["current_set"],
            "reps_done_in_set": session["reps_done_in_set"],
            "reps_per_set": session["reps_per_set"],
            "sets": session["sets"],
            "workout_complete": session["workout_complete"],
            "last_message": session.get("last_message", ""),
            "calories_burned": session.get("calories_burned", 0),
            "muscle_progress": session.get("muscle_progress", 0),
            "rep_history": session.get("rep_history", []),
            "calorie_history": session.get("calorie_history", []),
            "is_resting": session.get("is_resting", False)
        })
    else:
        return jsonify({"error": "No session found."}), 404


# ✅ Frame generator
def generate_frames(exercise):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam.")
        blank = 255 * np.ones((480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', blank)
        frame_bytes = buffer.tobytes()
        for _ in range(100):
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return

    print(f"🎥 Webcam stream started for: {exercise}")

    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Failed to read frame.")
            break

        try:
            # ✅ Call appropriate processor
            if exercise == "bicep_curl":
                frame, status, angle, reps = process_frame_curling(frame, pose)
            elif exercise == "squats":
                frame, status, angle, reps = process_frame_squats(frame, pose)
            elif exercise == "lunges":
                frame, status, angle, reps = process_frame_lunges(frame, pose)
            elif exercise == "tricep_kickback":
                frame, status, angle, reps = process_frame_tricep_kickback(frame, pose)
            else:
                reps = None

            if reps is not None:
                last_reps[exercise] = reps

            username = None
            for user, sess in user_sessions.items():
                if sess["exercise"] == exercise:
                    username = user
                    break

            if username and reps is not None:
                session_data = user_sessions[username]

                # ✅ NEW: skip updates while resting
                if session_data.get("is_resting", False):
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    continue

                reps_per_set = session_data["reps_per_set"]
                current_set = session_data["current_set"]
                total_sets = session_data["sets"]
                rest_time = session_data.get("rest_time", 60)

                reps_done_in_set = reps - ((current_set - 1) * reps_per_set)
                reps_done_in_set = max(reps_done_in_set, 0)

                prev_reps = session_data.get("prev_reps_done_in_set")
                prev_set = session_data.get("prev_set")

                if reps_done_in_set != prev_reps or current_set != prev_set:
                    reps_left = reps_per_set - reps_done_in_set
                    sets_left = total_sets - (current_set - 1)
                    print(f"👉 {username}: {reps_left} reps left in set {current_set}, {sets_left} sets remaining.")

                    session_data["prev_reps_done_in_set"] = reps_done_in_set
                    session_data["prev_set"] = current_set

                if reps_done_in_set >= reps_per_set:
                    session_data["last_message"] = f"✅ Completed set {current_set} for {exercise}."
                    session_data["current_set"] += 1
                    reps_done_in_set = 0

                    if session_data["current_set"] > total_sets:
                        session_data["workout_complete"] = True
                        session_data["last_message"] += f" 🎉 All sets complete for {exercise}!"
                        print(f"🎉 {username} has finished ALL sets for {exercise}!")
                    else:
                        session_data["last_message"] += f" 😴 Resting {rest_time} seconds before set {session_data['current_set']}..."
                        session_data["is_resting"] = True
                        cap.release()
                        time.sleep(rest_time)
                        cap = cv2.VideoCapture(0)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        session_data["is_resting"] = False
                        print(f"🔥 Resuming workout for set {session_data['current_set']}.")

                # ✅ Only update calories if not resting
                body_weight_kg = 70
                cal_per_rep = 6 * body_weight_kg * 3.5 / 200 * 0.033
                total_reps = ((current_set - 1) * reps_per_set) + reps_done_in_set
                calories_burned = round(total_reps * cal_per_rep, 2)

                session_data["calories_burned"] = calories_burned
                session_data["muscle_progress"] = total_reps
                session_data["reps_done_in_set"] = reps_done_in_set

                if not session_data.get("is_resting", False):
                    session_data["rep_history"].append({
                        "timestamp": time.time(),
                        "total_reps": total_reps
                    })

                    session_data["calorie_history"].append({
                        "timestamp": time.time(),
                        "calories": calories_burned
                    })

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"🔥 Frame processing error: {e}")
            break

    cap.release()
    print("📴 Webcam released.")


if __name__ == '__main__':
    print("🚀 Running on http://127.0.0.1:5000")
    app.run(debug=True)
