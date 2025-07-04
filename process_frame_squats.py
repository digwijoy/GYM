from utils import calculate_angle, get_pose_landmarks
import cv2

# Thresholds for angle classification
SQUAT_THRESHOLDS = {
    'down': 70,
    'up': 160
}

def process_frame_squats(frame, pose):
    """
    Process a video frame to detect squats and determine exercise status.

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
            print("🚫 No landmarks detected for squats.")
            return frame, None

        required_keys = ['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE']
        if not all(key in landmarks for key in required_keys):
            print("🚫 Missing required landmarks.")
            return frame, None

        hip = landmarks['LEFT_HIP']
        knee = landmarks['LEFT_KNEE']
        ankle = landmarks['LEFT_ANKLE']

        # Compute angle at the knee
        angle = calculate_angle(hip, knee, ankle)

        # Classify status based on angle
        if angle < SQUAT_THRESHOLDS['down']:
            status = "Squatting Down"
        elif angle > SQUAT_THRESHOLDS['up']:
            status = "Standing Up"
        else:
            status = "In Between"

        # Draw angle and status on the frame
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
        print(f"🔥 Error in process_frame_squats: {e}")
        return frame, None
