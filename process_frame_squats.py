from utils import calculate_angle, get_pose_landmarks
import cv2

# thresholds
SQUAT_DOWN_ANGLE = 90
SQUAT_UP_ANGLE = 160

# global counters
rep_count = 0
squat_state = "up"

def process_frame_squats(frame, pose):
    """
    Process a video frame to detect squats and determine exercise status.

    Args:
        frame (ndarray): BGR image from OpenCV
        pose (mp.solutions.pose.Pose): Mediapipe Pose object

    Returns:
        frame (ndarray): Frame with overlays (optional)
        status (str or None): Squat movement status
        angle (float or None): Measured knee angle
        reps (int or None): Total repetition count
    """
    global rep_count, squat_state

    try:
        landmarks, _ = get_pose_landmarks(frame, pose)
        if landmarks is None:
            print("🚫 No landmarks detected for squats.")
            return frame, None, None, None

        required_keys = ['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE']
        if not all(key in landmarks for key in required_keys):
            print("🚫 Missing required landmarks.")
            return frame, None, None, None

        hip = landmarks['LEFT_HIP']
        knee = landmarks['LEFT_KNEE']
        ankle = landmarks['LEFT_ANKLE']

        # Compute knee angle
        angle = calculate_angle(hip, knee, ankle)

        # Rep counting logic
        if angle < SQUAT_DOWN_ANGLE:
            if squat_state == "up":
                squat_state = "down"
        elif angle > SQUAT_UP_ANGLE:
            if squat_state == "down":
                squat_state = "up"
                rep_count += 1

        # Determine squat state
        if angle < SQUAT_DOWN_ANGLE:
            status = "Squatting Down"
        elif angle > SQUAT_UP_ANGLE:
            status = "Standing Up"
        else:
            status = "In Between"

        return frame, status, int(angle), rep_count

    except Exception as e:
        print(f"🔥 Error in process_frame_squats: {e}")
        return frame, None, None, None
