from ultralytics import YOLO
import os

data_yaml_file = os.path.join(os.getcwd(), "data.yaml")

model = YOLO('./vireon/weights/best.pt')


model.train(data=data_yaml_file, epochs=40, imgsz=640,  batch=8,  name="vireon", exist_ok=True, augment=True,
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    translate=0.05,
    scale=0.1,
    shear=0.0,
    perspective=0.0,
    mosaic=0.3,
    mixup=0.0,)

