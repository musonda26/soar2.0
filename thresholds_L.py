# thresholds for lunges

# thresholds.py

def get_thresholds_lunges_beginner():
    return {
        "knee_angle_min": 70,
        "knee_angle_max": 110,
        "hip_angle_min": 30,
        "hip_angle_max": 60,
        'CNT_FRAME_THRESH': 50,
        'INACTIVE_THRESH': 15.0,
    }

def get_thresholds_lunges_pro():
    return {
        "knee_angle_min": 80,
        "knee_angle_max": 100,
        "hip_angle_min": 40,
        "hip_angle_max": 50,
        'CNT_FRAME_THRESH': 50,
        'INACTIVE_THRESH': 15.0,
    }
