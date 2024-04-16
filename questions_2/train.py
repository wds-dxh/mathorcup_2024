from ultralytics import YOLO

#加载模型
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='mydata.yaml', epochs=100, imgsz=640)
