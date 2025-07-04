from utils import calculate_angle, get_pose_landmarks
import cv2

# Thresholds for angle classification
LUNGE_THRESHOLDS = {
    'down': 70,
    'up': 160
}

def process_frame_lunges(frame, pose):
    """
    Process a video frame to detect lunges and determine exercise status.

    Args:
        frame (ndarray): BGR image from OpenCV
        pose (mp.solutions.pose.Pose): Mediapipe Pose object

    Returns:
        frame (ndarray): Frame with overlaid text
        status (str or None): Status string if landmarks detected, else None
    """
    try:
        landmarks, _ = get_pose_landmarks(frame, pose)
        if landmarks is None:
            print("🚫 No landmarks detected for lunges.")
            return frame, None

        required_keys = ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE']
        if not all(key in landmarks for key in required_keys):
            print("🚫 Missing required landmarks for lunges.")
            return frame, None

        hip = landmarks['RIGHT_HIP']
        knee = landmarks['RIGHT_KNEE']
        ankle = landmarks['RIGHT_ANKLE']

        angle = calculate_angle(hip, knee, ankle)

        if angle < LUNGE_THRESHOLDS['down']:
            status = "Lunge Down"
        elif angle > LUNGE_THRESHOLDS['up']:
            status = "Standing Up"
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
        print(f"🔥 Error in process_frame_lunges: {e}")
        return frame, None
