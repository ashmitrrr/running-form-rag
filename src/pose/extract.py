from ultralytics import YOLO

model = YOLO("yolov8m-pose.pt")  # downloads on first run
results = model("data/raw/test_clip.mp4", save=True)  # save=True renders overlay