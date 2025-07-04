import numpy as np
import cv2
import mediapipe as mp

def calculate_angle(a: list, b: list, c: list) -> float:
    """
    Calculate the angle between three points.

    Args:
        a, b, c: Each a list [x, y] representing coordinates.

    Returns:
        angle: Angle in degrees between the three points.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


mp_pose = mp.solutions.pose
PoseLandmark = mp_pose.PoseLandmark

def get_pose_landmarks(frame, pose):
    """
    Extract pose landmarks from a frame using MediaPipe.

    Args:
        frame: Raw OpenCV frame (BGR).
        pose: An initialized mp_pose.Pose object.

    Returns:
        Tuple of:
            - Dictionary of landmark names mapped to [x, y] pixel positions
            - Full results object (optional use)
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return None, None

    h, w, _ = frame.shape
    landmarks = {}

    for i, landmark in enumerate(results.pose_landmarks.landmark):
        name = PoseLandmark(i).name
        landmarks[name] = [landmark.x * w, landmark.y * h]

    return landmarks, results


def draw_angle_arc(image, center, radius, start_angle, end_angle,
                   color=(255, 255, 0), thickness=2) -> None:
    """
    Draw an arc on the image representing the angle.

    Args:
        image: OpenCV image.
        center: Center of arc as (x, y).
        radius: Radius of the arc.
        start_angle: Starting angle in degrees.
        end_angle: Ending angle in degrees.
        color: Arc color in BGR.
        thickness: Arc line thickness.
    """
    axes = (radius, radius)
    angle = 0
    start_angle = int(start_angle)
    end_angle = int(end_angle)

    cv2.ellipse(
        image,
        tuple(map(int, center)),
        axes,
        angle,
        start_angle,
        end_angle,
        color,
        thickness
    )


def calculate_distance(point1: list, point2: list) -> float:
    """
    Compute Euclidean distance between two points.

    Args:
        point1, point2: Each a list [x, y].

    Returns:
        Euclidean distance.
    """
    return np.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


def normalize_landmarks(landmark, image_width: int, image_height: int) -> list:
    """
    Convert normalized MediaPipe landmark to pixel coordinates.

    Args:
        landmark: A MediaPipe landmark.
        image_width: Width of the frame.
        image_height: Height of the frame.

    Returns:
        [x, y] pixel coordinates.
    """
    return [
        int(landmark.x * image_width),
        int(landmark.y * image_height)
    ]


def get_visibility_score(landmarks, required_landmarks: list) -> float:
    """
    Compute average visibility score for a list of landmarks.

    Args:
        landmarks: MediaPipe landmarks object.
        required_landmarks: List of landmark indices.

    Returns:
        Average visibility score between 0 and 1.
    """
    if not landmarks:
        return 0.0

    total_visibility = 0.0
    for idx in required_landmarks:
        if idx < len(landmarks.landmark):
            total_visibility += landmarks.landmark[idx].visibility

    return total_visibility / len(required_landmarks)


def check_pose_quality(landmarks, mp_pose, exercise_type: str) -> tuple:
    """
    Check if the pose quality is acceptable for exercise detection.

    Args:
        landmarks: MediaPipe landmarks object.
        mp_pose: MediaPipe pose module.
        exercise_type: The exercise being performed.

    Returns:
        Tuple of:
            is_good_quality: Boolean indicating if quality is acceptable.
            feedback: Feedback message.
    """
    if not landmarks:
        return False, "No pose detected"

    required_landmarks = {
        'bicep_curl': [
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_WRIST.value
        ],
        'squats': [
            mp_pose.PoseLandmark.LEFT_HIP.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value,
            mp_pose.PoseLandmark.LEFT_ANKLE.value
        ],
        'lunges': [
            mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_KNEE.value,
            mp_pose.PoseLandmark.RIGHT_ANKLE.value
        ],
        'tricep_kickback': [
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_WRIST.value
        ],
        'dumbbell_fly': [
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value
        ]
    }

    if exercise_type not in required_landmarks:
        return True, "Exercise type not recognized"

    visibility_score = get_visibility_score(
        landmarks,
        required_landmarks[exercise_type]
    )

    if visibility_score < 0.5:
        return False, "Poor pose visibility - adjust your position"
    elif visibility_score < 0.7:
        return True, "Pose quality is acceptable"
    else:
        return True, "Excellent pose quality"


def format_time(seconds: float) -> str:
    """
    Format seconds as MM:SS string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string.
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def apply_smoothing(current_value: float,
                    previous_value: float,
                    smoothing_factor: float = 0.8) -> float:
    """
    Smooth values to reduce noise.

    Args:
        current_value: New measurement.
        previous_value: Last measurement.
        smoothing_factor: Smoothing factor (0-1).

    Returns:
        Smoothed value.
    """
    if previous_value is None:
        return current_value

    return smoothing_factor * previous_value + \
           (1 - smoothing_factor) * current_value


