from utils import calculate_angle, get_pose_landmarks
from thresholds import CURL_THRESHOLDS
import cv2

def process_frame_curling(frame, pose):
    """
    Process a video frame for bicep curl detection.

    Args:
        frame (ndarray): BGR frame from OpenCV
        pose (mp.solutions.pose.Pose): Mediapipe Pose object

    Returns:
        frame (ndarray): Annotated frame
        status (str or None): Detected status (e.g. "Curl Up", "Arm Down"), or None if landmarks not found.
    """
    try:
        landmarks, _ = get_pose_landmarks(frame, pose)
        if landmarks is None:
            print("🚫 No landmarks detected for bicep curl.")
            return frame, None

        required_keys = ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']
        if not all(k in landmarks for k in required_keys):
            print("🚫 Missing right arm landmarks.")
            return frame, None

        shoulder = landmarks['RIGHT_SHOULDER']
        elbow = landmarks['RIGHT_ELBOW']
        wrist = landmarks['RIGHT_WRIST']

        angle = calculate_angle(shoulder, elbow, wrist)

        if angle < CURL_THRESHOLDS['up']:
            status = "Curl Up"
        elif angle > CURL_THRESHOLDS['down']:
            status = "Arm Down"
        else:
            status = "In Between"

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

        return frame, status

    except Exception as e:
        print(f"🔥 Error in process_frame_curling: {e}")
        return frame, None
