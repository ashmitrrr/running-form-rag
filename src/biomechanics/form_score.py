import pandas as pd
import numpy as np
from scipy.signal import find_peaks

FPS = 30.0

# which metrics feed the form score, and how much each matters
# Weights sum to 1.0 and tuning
METRIC_WEIGHTS = {
    "cadence": 0.25,
    "trunk_lean": 0.20,
    "knee_angle": 0.20,
    "vertical_oscillation": 0.20,
    "arm_symmetry": 0.15,
}


def _angle_from_vertical(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    return np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))


def _joint_angle(ax, ay, bx, by, cx, cy):
    ba = np.array([ax - bx, ay - by])
    bc = np.array([cx - bx, cy - by])
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def compute_windowed_metrics(df, fps, window_sec=5.0):
    """
    Compute the 5 metrics per rolling time window, returning one row per window
    gives a time series of form, and not just one number for the whole run
    """
    frames_per_window = int(window_sec * fps)
    n_windows = len(df) // frames_per_window

    rows = []
    for w in range(n_windows):
        start = w * frames_per_window
        end = start + frames_per_window
        chunk = df.iloc[start:end].reset_index(drop=True)

        row = {"window": w, "t_start_sec": round(start / fps, 1)}

        #cadence (per window)
        total_steps = 0
        for side in ["left", "right"]:
            y = chunk[f"{side}_ankle_y"].values
            y_s = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values
            peaks, _ = find_peaks(y_s, distance=10)
            total_steps += len(peaks)
        window_min = (len(chunk) / fps) / 60
        row["cadence"] = total_steps / window_min if window_min > 0 else 0

        #trunk lean
        sh_x = (chunk["left_shoulder_x"] + chunk["right_shoulder_x"]) / 2
        sh_y = (chunk["left_shoulder_y"] + chunk["right_shoulder_y"]) / 2
        hip_x = (chunk["left_hip_x"] + chunk["right_hip_x"]) / 2
        hip_y = (chunk["left_hip_y"] + chunk["right_hip_y"]) / 2
        row["trunk_lean"] = float(_angle_from_vertical(hip_x, hip_y, sh_x, sh_y).mean())

        #knee angle (mean over window)
        knee_angles = []
        for f in range(len(chunk)):
            knee_angles.append(_joint_angle(
                chunk["left_hip_x"].iloc[f], chunk["left_hip_y"].iloc[f],
                chunk["left_knee_x"].iloc[f], chunk["left_knee_y"].iloc[f],
                chunk["left_ankle_x"].iloc[f], chunk["left_ankle_y"].iloc[f],
            ))
        row["knee_angle"] = float(np.mean(knee_angles))

        #vertical oscillation
        hip_mid_y = hip_y.values
        hip_s = pd.Series(hip_mid_y).rolling(3, center=True, min_periods=1).mean().values
        row["vertical_oscillation"] = float(np.percentile(hip_s, 95) - np.percentile(hip_s, 5))

        #arm symmetry
        arm = {}
        for side in ["left", "right"]:
            a = [ _joint_angle(
                    chunk[f"{side}_shoulder_x"].iloc[f], chunk[f"{side}_shoulder_y"].iloc[f],
                    chunk[f"{side}_elbow_x"].iloc[f], chunk[f"{side}_elbow_y"].iloc[f],
                    chunk[f"{side}_wrist_x"].iloc[f], chunk[f"{side}_wrist_y"].iloc[f],
                ) for f in range(len(chunk)) ]
            arm[side] = np.mean(a)
        row["arm_symmetry"] = float(abs(arm["left"] - arm["right"]))

        rows.append(row)

    return pd.DataFrame(rows)


def compute_form_score(keypoints_path, fps=FPS, window_sec=5.0,
                       baseline="first_n", baseline_windows=3):
    """
    Relative form score per window.
    Each metric is scored by how far it drifts from the runner's own baseline.
    100 = identical to baseline; lower = more deviation from baseline form.
    """
    df = pd.read_parquet(keypoints_path)
    wm = compute_windowed_metrics(df, fps, window_sec)

    metric_cols = list(METRIC_WEIGHTS.keys())

    # establish baseline value per metric
    if baseline == "first_n":
        base = wm[metric_cols].iloc[:baseline_windows].mean()
    else:  # "median"
        base = wm[metric_cols].median()

    # for each window, score each metric as % deviation from baseline
    # convert to a 0-100 sub-score: 100 that is no deviation
    for col in metric_cols:
        b = base[col] if base[col] != 0 else 1e-9
        pct_dev = (wm[col] - b).abs() / abs(b)        # fractional deviation
        wm[f"{col}_score"] = (100 * (1 - pct_dev)).clip(lower=0)  # 0-100, floored at 0

    # weighted combine into overall form score per window
    wm["form_score"] = sum(
        wm[f"{col}_score"] * METRIC_WEIGHTS[col] for col in metric_cols
    ).round(1)

    return wm


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/biomechanics/form_score.py <keypoints_parquet> [fps] [window_sec]")
        sys.exit(1)

    path = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) >= 3 else FPS
    window = float(sys.argv[3]) if len(sys.argv) >= 4 else 5.0

    result = compute_form_score(path, fps=fps, window_sec=window)
    cols = ["window", "t_start_sec", "form_score"] + [f"{m}_score" for m in METRIC_WEIGHTS]
    print(result[cols].to_string(index=False))