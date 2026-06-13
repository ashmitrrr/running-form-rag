import pandas as pd
import numpy as np
from scipy.signal import find_peaks

FPS = 30.0

def compute_cadence(keypoints_path: str, fps: float = FPS):
    df = pd.read_parquet(keypoints_path)

    # y increases DOWNWARD. So a footstrike is PEAK in the y values, and mid-swing (ankle highest) is a trough (image coords)
    # we detect peaks on each ankle separately, then combine

    results = {}
    total_steps = 0

    for side in ["left", "right"]:
        y = df[f"{side}_ankle_y"].values

        # smooth a little to kill single frame jitters before peak detection
        y_smooth = pd.Series(y).rolling(window=3, center=True, min_periods=1).mean().values

        # distance: min frames between two footstrikes of the same foot
        # runner won't exceed ~3 strides/sec per foot, at 30fps and
        # ~10 frames minimum between same foot strikes, also keep double counting out
        peaks, _ = find_peaks(y_smooth, distance=10)

        steps = len(peaks)
        total_steps += steps
        results[f"{side}_steps"] = steps

    duration_sec = len(df) / fps
    duration_min = duration_sec / 60

    # total_steps counts both feet = total footstrikes = total steps
    cadence_spm = total_steps / duration_min

    results["duration_sec"] = round(duration_sec, 2)
    results["total_steps"] = total_steps
    results["cadence_spm"] = round(cadence_spm, 1)

    return results

if __name__ == "__main__":
    out = compute_cadence("data/processed/test_clip_keypoints.parquet")
    print(out)