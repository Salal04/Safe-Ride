from ultralytics import YOLO
import os

# model load
model = YOLO("E:\\dataset\\projectCode\\deepSort\\mirror-indicator-yolov10m\\weights\\bestM.pt")
# input & output folders
input_folder = "E:\\dataset\\projectCode\\deepSort\\test\\testimage"
output_project = "E:\\dataset\\projectCode\\deepSort\\mirror-indicator-yolov10m"
run_name = "run_2"

for img in os.listdir(input_folder):
    if img.lower().endswith((".jpg", ".png", ".jpeg")):
        model.predict(
            source=os.path.join(input_folder, img),
            imgsz=640,
            conf=0.4,
            save=True,
            project=output_project,
            name=run_name
        )


