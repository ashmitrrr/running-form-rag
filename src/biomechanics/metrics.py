import pandas as pd
import numpy as np
from scipy.signal import find_peaks

FPS = 30.0


def _angle_from_vertical(x1, y1, x2, y2):
    """
    Angle (degrees) of the line from point1->point2 relative to vertical,
    0 = perfectly upright, larger = more leaning.
    """
    dx = x2 - x1
    dy = y2 - y1
    # angle from vertical axis and abs lean, magnitude regardless of direction
    angle = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
    return angle


def _joint_angle(ax, ay, bx, by, cx, cy):
    """
    Interior angle (degrees) at joint B, formed by points A-B-C.
    e.g. knee angle = angle at knee formed by hip-knee-ankle.
    """
    ba = np.array([ax - bx, ay - by])
    bc = np.array([cx - bx, cy - by])
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def compute_metrics(keypoints_path: str, fps: float = FPS):
    df = pd.read_parquet(keypoints_path)
    results = {}

    # cadence
    total_steps = 0
    for side in ["left", "right"]:
        y = df[f"{side}_ankle_y"].values
        y_smooth = pd.Series(y).rolling(window=3, center=True, min_periods=1).mean().values
        peaks, _ = find_peaks(y_smooth, distance=10)
        results[f"{side}_steps"] = len(peaks)
        total_steps += len(peaks)

    duration_min = (len(df) / fps) / 60
    results["cadence_spm"] = round(total_steps / duration_min, 1)

    #trunk lean
    # use midpoints of left/right shoulder and left/right hip for stability
    sh_x = (df["left_shoulder_x"] + df["right_shoulder_x"]) / 2
    sh_y = (df["left_shoulder_y"] + df["right_shoulder_y"]) / 2
    hip_x = (df["left_hip_x"] + df["right_hip_x"]) / 2
    hip_y = (df["left_hip_y"] + df["right_hip_y"]) / 2

    trunk_angles = _angle_from_vertical(hip_x, hip_y, sh_x, sh_y)
    results["trunk_lean_deg"] = round(float(trunk_angles.mean()), 1)
