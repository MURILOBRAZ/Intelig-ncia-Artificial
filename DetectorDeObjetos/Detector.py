from ultralytics import YOLO

modelo = YOLO('yolov8n.pt')

modelo.predict(source='1', show=True)
