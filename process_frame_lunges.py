from utils import calculate_angle, get_pose_landmarks
import cv2

# Thresholds for angle classification
LUNGE_THRESHOLDS = {
    'down': 70,
    'up': 160
}

# Global state for rep counting
lunge_rep_count = 0
lunge_state = "up"  # assume starting from standing position

def process_frame_lunges(frame, pose):
    """
    Process a video frame to detect lunges and count reps.

    Args:
        frame (ndarray): BGR image from OpenCV
        pose (mp.solutions.pose.Pose): Mediapipe Pose object

    Returns:
        frame (ndarray): Frame with overlays
        status (str or None): Movement status
        angle (float or None): Measured knee angle
        reps (int or None): Total repetition count
    """
    global lunge_rep_count, lunge_state

    try:
        landmarks, _ = get_pose_landmarks(frame, pose)
        if landmarks is None:
            print("🚫 No landmarks detected for lunges.")
            return frame, None, None, None

        required_keys = ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE']
        if not all(key in landmarks for key in required_keys):
            print("🚫 Missing required landmarks for lunges.")
            return frame, None, None, None

        hip = landmarks['RIGHT_HIP']
        knee = landmarks['RIGHT_KNEE']
        ankle = landmarks['RIGHT_ANKLE']

        # Calculate the knee angle
        angle = calculate_angle(hip, knee, ankle)

        # Rep counting logic
        if angle < LUNGE_THRESHOLDS['down']:
            if lunge_state == "up":
                lunge_state = "down"
        elif angle > LUNGE_THRESHOLDS['up']:
            if lunge_state == "down":
                lunge_state = "up"
                lunge_rep_count += 1

        # Status classification
        if angle < LUNGE_THRESHOLDS['down']:
            status = "Lunge Down"
        elif angle > LUNGE_THRESHOLDS['up']:
            status = "Standing Up"
        else:
            status = "In Between"

        # Overlay on frame
        cv2.putText(frame, f'Angle: {int(angle)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Status: {status}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Reps: {lunge_rep_count}', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return frame, status, int(angle), lunge_rep_count

    except Exception as e:
        print(f"🔥 Error in process_frame_lunges: {e}")
        return frame, None, None, None
