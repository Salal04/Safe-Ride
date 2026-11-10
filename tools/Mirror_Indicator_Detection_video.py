from ultralytics import YOLO
from config import *
import os

# Model load
model = YOLO(str(INDICATOR_MODEL_PATH))

# Input & output
input_video = str(LEGACY_VIDEO_PATH)
output_project = str(LEGACY_OUTPUT_DIR)
run_name = "video_run_2"

# Video prediction
results = model.predict(
    source=input_video,  # video file path
    imgsz=LEGACY_IMAGE_SIZE,
    conf=LEGACY_DETECTOR_CONFIDENCE,
    save=True,           # save output
    project=output_project,
    name=run_name
)

print(f"✅ Video saved in: {os.path.join(output_project, run_name)}")
