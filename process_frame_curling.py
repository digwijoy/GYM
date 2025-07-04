from utils import calculate_angle, get_pose_landmarks
from thresholds import CURL_THRESHOLDS
import cv2

# Thresholds
CURL_UP_THRESHOLD = CURL_THRESHOLDS['up']
CURL_DOWN_THRESHOLD = CURL_THRESHOLDS['down']

# Global counters
curl_rep_count = 0
curl_state = "down"  # assume starting with arm down

def process_frame_curling(frame, pose):
    """
    Process a video frame for bicep curl detection and rep counting.

    Args:
        frame (ndarray): BGR frame from OpenCV
        pose (mp.solutions.pose.Pose): Mediapipe Pose object

    Returns:
        frame (ndarray): Annotated frame
        status (str or None): Movement status
        angle (float or None): Elbow angle measured
        reps (int or None): Total repetition count
    """
    global curl_rep_count, curl_state

    try:
        landmarks, _ = get_pose_landmarks(frame, pose)
        if landmarks is None:
            print("🚫 No landmarks detected for bicep curl.")
            return frame, None, None, None

        required_keys = ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']
        if not all(k in landmarks for k in required_keys):
            print("🚫 Missing right arm landmarks.")
            return frame, None, None, None

        shoulder = landmarks['RIGHT_SHOULDER']
        elbow = landmarks['RIGHT_ELBOW']
        wrist = landmarks['RIGHT_WRIST']

        angle = calculate_angle(shoulder, elbow, wrist)

        # Rep counting logic
        if angle < CURL_UP_THRESHOLD:
            if curl_state == "down":
                curl_state = "up"
        elif angle > CURL_DOWN_THRESHOLD:
            if curl_state == "up":
                curl_state = "down"
                curl_rep_count += 1

        # Determine movement status
        if angle < CURL_UP_THRESHOLD:
            status = "Curl Up"
        elif angle > CURL_DOWN_THRESHOLD:
            status = "Arm Down"
        else:
            status = "In Between"

        # Overlay data on frame
        cv2.putText(
            frame, f'Angle: {int(angle)}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2, cv2.LINE_AA
        )

        cv2.putText(
            frame, f'Status: {status}',
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 0), 2, cv2.LINE_AA
        )

        cv2.putText(
            frame, f'Reps: {curl_rep_count}',
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2, cv2.LINE_AA
        )

        return frame, status, int(angle), curl_rep_count

    except Exception as e:
        print(f"🔥 Error in process_frame_curling: {e}")
        return frame, None, None, None
