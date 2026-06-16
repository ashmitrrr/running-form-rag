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

    #knee angle (footstrike)
    # compute knee angle each frame, then average it specifically at footstrike frames
    # use the side with stronger ankle confidence as the "lead" leg for footstrike timing
    knee_angles_at_strike = []
    for side in ["left", "right"]:
        ankle_y = df[f"{side}_ankle_y"].values
        ankle_y_smooth = pd.Series(ankle_y).rolling(window=3, center=True, min_periods=1).mean().values
        strike_frames, _ = find_peaks(ankle_y_smooth, distance=10)

        for f in strike_frames:
            angle = _joint_angle(
                df[f"{side}_hip_x"].iloc[f], df[f"{side}_hip_y"].iloc[f],
                df[f"{side}_knee_x"].iloc[f], df[f"{side}_knee_y"].iloc[f],
                df[f"{side}_ankle_x"].iloc[f], df[f"{side}_ankle_y"].iloc[f],
            )
            knee_angles_at_strike.append(angle)

    results["knee_angle_footstrike_deg"] = round(float(np.mean(knee_angles_at_strike)), 1)

    # VO
    hip_mid_y = hip_y.values
    hip_mid_y_smooth = pd.Series(hip_mid_y).rolling(window=3, center=True, min_periods=1).mean().values
    # use the std-based amplitude: peak-to-trough spread of vertical position
    oscillation = (np.percentile(hip_mid_y_smooth, 95) - np.percentile(hip_mid_y_smooth, 5))
    results["vertical_oscillation_px"] = round(float(oscillation), 1)

    # arm swing
    # elbow angle (shoulder elbow wrist) for each arm, averaged and report
    # smaller delta = more symmetric arm swing
    arm_angles = {}
    for side in ["left", "right"]:
        angles = []
        for f in range(len(df)):
            angle = _joint_angle(
                df[f"{side}_shoulder_x"].iloc[f], df[f"{side}_shoulder_y"].iloc[f],
                df[f"{side}_elbow_x"].iloc[f], df[f"{side}_elbow_y"].iloc[f],
                df[f"{side}_wrist_x"].iloc[f], df[f"{side}_wrist_y"].iloc[f],
            )
            angles.append(angle)
        arm_angles[side] = np.mean(angles)

    results["arm_symmetry_delta_deg"] = round(float(abs(arm_angles["left"] - arm_angles["right"])), 1)

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/biomechanics/metrics.py <keypoints_parquet> [fps]")
        print("Example: python src/biomechanics/metrics.py data/processed/onspot_run_keypoints.parquet 29.88")
        sys.exit(1)

    keypoints_path = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) >= 3 else FPS

    out = compute_metrics(keypoints_path, fps=fps)
    for k, v in out.items():
        print(f"{k}: {v}")