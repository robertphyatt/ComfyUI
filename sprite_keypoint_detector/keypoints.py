"""Keypoint definitions for sprite skeleton detection."""

KEYPOINT_NAMES = [
    "head",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_fingertip",   # tip of left hand/fist
    "right_fingertip",  # tip of right hand/fist
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_toe",         # tip of left foot
    "right_toe",        # tip of right foot
]

NUM_KEYPOINTS = len(KEYPOINT_NAMES)

# Skeleton connections for visualization (pairs of keypoint indices)
# Indices: 0-1 head/neck, 2-7 arms, 8-9 fingertips, 10-13 hips/knees, 14-15 ankles, 16-17 toes
SKELETON_CONNECTIONS = [
    (0, 1),   # head -> neck
    (1, 2),   # neck -> left_shoulder
    (1, 3),   # neck -> right_shoulder
    (2, 4),   # left_shoulder -> left_elbow
    (3, 5),   # right_shoulder -> right_elbow
    (4, 6),   # left_elbow -> left_wrist
    (5, 7),   # right_elbow -> right_wrist
    (6, 8),   # left_wrist -> left_fingertip
    (7, 9),   # right_wrist -> right_fingertip
    (1, 10),  # neck -> left_hip (via spine, simplified)
    (1, 11),  # neck -> right_hip (via spine, simplified)
    (10, 11), # left_hip -> right_hip
    (10, 12), # left_hip -> left_knee
    (11, 13), # right_hip -> right_knee
    (12, 14), # left_knee -> left_ankle
    (13, 15), # right_knee -> right_ankle
    (14, 16), # left_ankle -> left_toe
    (15, 17), # right_ankle -> right_toe
]

# Colors for each limb segment (RGB for visualization)
SKELETON_COLORS = [
    (255, 255, 255),  # head-neck: white
    (0, 255, 0),      # neck-left_shoulder: green
    (255, 165, 0),    # neck-right_shoulder: orange
    (0, 255, 0),      # left_shoulder-left_elbow: green
    (255, 165, 0),    # right_shoulder-right_elbow: orange
    (0, 255, 255),    # left_elbow-left_wrist: cyan
    (255, 0, 255),    # right_elbow-right_wrist: magenta
    (0, 255, 255),    # left_wrist-left_fingertip: cyan
    (255, 0, 255),    # right_wrist-right_fingertip: magenta
    (0, 255, 0),      # neck-left_hip: green
    (255, 165, 0),    # neck-right_hip: orange
    (255, 0, 0),      # left_hip-right_hip: red
    (0, 255, 0),      # left_hip-left_knee: green
    (255, 165, 0),    # right_hip-right_knee: orange
    (0, 128, 255),    # left_knee-left_ankle: light blue
    (255, 128, 0),    # right_knee-right_ankle: dark orange
    (0, 128, 255),    # left_ankle-left_toe: light blue
    (255, 128, 0),    # right_ankle-right_toe: dark orange
]
