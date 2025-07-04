# ================================
# Thresholds and Parameters
# ================================

# ------------------------------
# Bicep Curl Thresholds
# ------------------------------
CURL_THRESHOLDS = {
    'up': 50,            # Angle when arm is curled up
    'down': 160,         # Angle when arm is extended down
    'min_angle': 30,     # Minimum angle for a valid curl
    'max_angle': 170     # Maximum angle for a valid curl
}

# ------------------------------
# Squat Thresholds
# ------------------------------
SQUAT_THRESHOLDS = {
    'up': 170,           # Angle when standing up
    'down': 90,          # Angle when in squat position
    'min_angle': 70,     # Minimum angle for a valid squat
    'max_angle': 180     # Maximum angle when standing
}

# ------------------------------
# Lunge Thresholds
# ------------------------------
LUNGE_THRESHOLDS = {
    'up': 170,           # Angle when standing up
    'down': 90,          # Angle when in lunge position
    'min_angle': 70,     # Minimum angle for a valid lunge
    'max_angle': 180     # Maximum angle when standing
}

# ------------------------------
# Tricep Kickback Thresholds
# ------------------------------
TRICEP_THRESHOLDS = {
    'extended': 160,     # Angle when arm is extended back
    'contracted': 90,    # Angle when arm is contracted
    'min_angle': 80,     # Minimum angle for valid movement
    'max_angle': 170     # Maximum angle for valid movement
}

# ------------------------------
# Dumbbell Fly Thresholds
# ------------------------------
FLY_THRESHOLDS = {
    'extended': 160,     # Angle when arms are extended out
    'contracted': 90,    # Angle when arms are brought together
    'min_angle': 80,     # Minimum angle for valid movement
    'max_angle': 170     # Maximum angle for valid movement
}

# ------------------------------
# Push-Up Thresholds
# ------------------------------
PUSHUP_THRESHOLDS = {
    'up': 160,           # Angle when arms are extended
    'down': 90,          # Angle when in push-up position
    'min_angle': 70,     # Minimum angle for a valid push-up
    'max_angle': 180     # Maximum angle when arms extended
}

# ------------------------------
# General Exercise Settings
# ------------------------------
EXERCISE_SETTINGS = {
    'bicep_curl': {
        'name': 'Bicep Curl',
        'description': 'Curl your arm up and down with controlled movement',
        'target_reps': 10,
        'rest_time': 60,          # seconds
        'muscle_groups': ['Biceps', 'Forearms'],
        'difficulty': 'Beginner'
    },
    'squats': {
        'name': 'Squats',
        'description': 'Lower your body by bending your knees and hips',
        'target_reps': 15,
        'rest_time': 90,
        'muscle_groups': ['Quadriceps', 'Glutes', 'Hamstrings'],
        'difficulty': 'Beginner'
    },
    'lunges': {
        'name': 'Lunges',
        'description': 'Step forward and lower your body until both knees are bent',
        'target_reps': 10,
        'rest_time': 60,
        'muscle_groups': ['Quadriceps', 'Glutes', 'Hamstrings', 'Calves'],
        'difficulty': 'Intermediate'
    },
    'tricep_kickback': {
        'name': 'Tricep Kickback',
        'description': 'Extend your arm back while keeping your elbow stationary',
        'target_reps': 12,
        'rest_time': 60,
        'muscle_groups': ['Triceps'],
        'difficulty': 'Intermediate'
    },
    'dumbbell_fly': {
        'name': 'Dumbbell Fly',
        'description': 'Bring your arms together in a wide arc motion',
        'target_reps': 10,
        'rest_time': 90,
        'muscle_groups': ['Chest', 'Shoulders'],
        'difficulty': 'Intermediate'
    }
}

# ------------------------------
# Pose Quality Thresholds
# ------------------------------
POSE_QUALITY = {
    'excellent': 0.9,
    'good': 0.7,
    'acceptable': 0.5,
    'poor': 0.3
}

# ------------------------------
# Timing Settings
# ------------------------------
TIMING_SETTINGS = {
    'hold_time': 0.5,            # Time to hold position for rep confirmation
    'min_rep_time': 1.0,         # Minimum time between reps
    'max_rep_time': 10.0,        # Maximum time between reps
    'session_timeout': 300       # Session timeout in seconds
}

# ------------------------------
# Feedback Messages
# ------------------------------
FEEDBACK_MESSAGES = {
    'good_rep': [
        "Excellent form!",
        "Great rep!",
        "Perfect execution!",
        "Well done!",
        "Keep it up!"
    ],
    'improve_form': [
        "Focus on your form",
        "Control the movement",
        "Slow and steady",
        "Mind your posture",
        "Keep your core engaged"
    ],
    'range_of_motion': [
        "Full range of motion",
        "Go deeper",
        "Complete the movement",
        "Extend fully",
        "Don't rush"
    ],
    'positioning': [
        "Adjust your position",
        "Face the camera",
        "Step back a bit",
        "Make sure you're visible",
        "Center yourself"
    ]
}

# ------------------------------
# Color Scheme for UI
# ------------------------------
COLORS = {
    'primary': (0, 255, 0),         # Green
    'secondary': (255, 255, 0),     # Yellow
    'accent': (255, 0, 255),        # Magenta
    'warning': (0, 165, 255),       # Orange
    'error': (0, 0, 255),           # Red
    'text': (255, 255, 255),        # White
    'background': (0, 0, 0)         # Black
}

# ------------------------------
# Exercise Progression Levels
# ------------------------------
PROGRESSION_LEVELS = {
    'beginner': {
        'reps': 8,
        'sets': 2,
        'rest_time': 90
    },
    'intermediate': {
        'reps': 12,
        'sets': 3,
        'rest_time': 60
    },
    'advanced': {
        'reps': 15,
        'sets': 4,
        'rest_time': 45
    }
}

# ------------------------------
# Workout Programs
# ------------------------------
WORKOUT_PROGRAMS = {
    'upper_body': {
        'exercises': ['bicep_curl', 'tricep_kickback', 'dumbbell_fly'],
        'duration': 20,     # minutes
        'difficulty': 'Intermediate'
    },
    'lower_body': {
        'exercises': ['squats', 'lunges'],
        'duration': 15,     # minutes
        'difficulty': 'Beginner'
    },
    'full_body': {
        'exercises': ['bicep_curl', 'squats', 'lunges', 'tricep_kickback', 'dumbbell_fly'],
        'duration': 30,     # minutes
        'difficulty': 'Advanced'
    }
}
