from ultralytics import YOLO
from config import *

model = YOLO(str(MIRROR_MODEL_PATH))
model.export(format="onnx")

model = YOLO(str(INDICATOR_MODEL_PATH))
model.export(format="onnx")
