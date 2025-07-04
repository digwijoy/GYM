from utils import calculate_angle, get_pose_landmarks
import cv2
from thresholds import TRICEP_THRESHOLDS

# Thresholds
CONTRACTED_THRESHOLD = TRICEP_THRESHOLDS['contracted']
EXTENDED_THRESHOLD = TRICEP_THRESHOLDS['extended']

# Global counters
tricep_rep_count = 0
tricep_state = "extended"   # start assuming arm is extended

def process_frame_tricep_kickback(frame, pose):
    """
    Process a video frame to detect tricep kickbacks and count reps.

    Args:
        frame (ndarray): BGR image from OpenCV
        pose (mp.solutions.pose.Pose): Mediapipe Pose object

    Returns:
        frame (ndarray): Frame with overlays
        status (str or None): Movement status
        angle (float or None): Measured elbow angle
        reps (int or None): Total repetition count
    """
    global tricep_rep_count, tricep_state

    try:
        landmarks, _ = get_pose_landmarks(frame, pose)
        if landmarks is None:
            print("🚫 No landmarks detected for tricep kickback.")
            return frame, None, None, None

        required_keys = ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']
        if not all(key in landmarks for key in required_keys):
            print("🚫 Missing required landmarks.")
            return frame, None, None, None

        shoulder = landmarks['RIGHT_SHOULDER']
        elbow = landmarks['RIGHT_ELBOW']
        wrist = landmarks['RIGHT_WRIST']

        angle = calculate_angle(shoulder, elbow, wrist)

        # Rep counting logic
        if angle < CONTRACTED_THRESHOLD:
            if tricep_state == "extended":
                tricep_state = "contracted"
        elif angle > EXTENDED_THRESHOLD:
            if tricep_state == "contracted":
                tricep_state = "extended"
                tricep_rep_count += 1

        # Determine status
        if angle < CONTRACTED_THRESHOLD:
            status = "Arm Contracted"
        elif angle > EXTENDED_THRESHOLD:
            status = "Arm Extended"
        else:
            status = "Mid-motion"

        # Overlay text on frame
        cv2.putText(frame, f'Angle: {int(angle)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Status: {status}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'Reps: {tricep_rep_count}', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame, status, int(angle), tricep_rep_count

    except Exception as e:
        print(f"🔥 Error in process_frame_tricep_kickback: {e}")
        return frame, None, None, None
