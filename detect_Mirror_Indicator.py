from ultralytics import YOLO
model = YOLO("mirror-indicator-yolov10m\\weights\\best.pt")


result = model('img1.jpg')

print(result)

