from ultralytics import YOLO
from config import *
import os

# model load
model = YOLO(str(MIRROR_MODEL_PATH))
# input & output folders
input_folder = str(LEGACY_TEST_IMAGE_DIR)
output_project = str(LEGACY_OUTPUT_DIR)
run_name = "run_2"

for img in os.listdir(input_folder):
    if img.lower().endswith((".jpg", ".png", ".jpeg")):
        model.predict(
            source=os.path.join(input_folder, img),
            imgsz=LEGACY_IMAGE_SIZE,
            conf=LEGACY_DETECTOR_CONFIDENCE,
            save=True,
            project=output_project,
            name=run_name
        )


