from ultralytics import YOLO

model = YOLO("mirror-indicator-yolov10m/weights/bestM.pt")
model.export(format="onnx")

model = YOLO("mirror-indicator-yolov10m/weights/bestI.pt")
model.export(format="onnx")
