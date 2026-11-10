from ultralytics import YOLO
import os

# Model load
model = YOLO("E:\\dataset\\projectCode\\deepSort\\mirror-indicator-yolov10m\\weights\\bestI.pt")

# Input & output
input_video = "E:\\dataset\\projectCode\\deepSort\\vid1.mp4"
output_project = "E:\\dataset\\projectCode\\deepSort\\mirror-indicator-yolov10m"
run_name = "video_run_2"

# Video prediction
results = model.predict(
    source=input_video,  # video file path
    imgsz=640,
    conf=0.4,
    save=True,           # save output
    project=output_project,
    name=run_name
)

print(f"✅ Video saved in: {os.path.join(output_project, run_name)}")
