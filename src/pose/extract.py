from ultralytics import YOLO
import numpy as np
import pandas as pd
from pathlib import Path

# COCO 17 keypoint names, in the order YOLOv8-pose outputs them
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

def extract_keypoints(video_path: str, output_path: str):
    model = YOLO("yolov8m-pose.pt")
    results = model(video_path, stream=True)  # stream=True = memory-efficient, one frame at a time

    rows = []
    for frame_idx, result in enumerate(results):
        if result.keypoints is None or len(result.keypoints) == 0:
            continue  # no person detected this frame

        # take the first and only person
        kpts = result.keypoints.data[0].cpu().numpy()  # shape: (17, 3) -> x, y, confidence

        row = {"frame": frame_idx}
        for i, name in enumerate(KEYPOINT_NAMES):
            row[f"{name}_x"] = kpts[i][0]
            row[f"{name}_y"] = kpts[i][1]
            row[f"{name}_conf"] = kpts[i][2]
        rows.append(row)

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} frames to {output_path}")
    return df

if __name__ == "__main__":
    extract_keypoints(
        "data/raw/test_clip.mp4",
        "data/processed/test_clip_keypoints.parquet",
    )