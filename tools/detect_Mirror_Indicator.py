from ultralytics import YOLO
from config import *
model = YOLO(str(MIRROR_MODEL_PATH))


result = model(str(LEGACY_IMAGE_PATH))

print(result)

