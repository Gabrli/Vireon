from ultralytics import YOLO
import os

model = YOLO('C:/Users/pawel/runs/detect/my_exp/weights/best.pt')

# data_yaml_file = os.path.join(os.getcwd(), "data.yaml")

# model.train(data=data_yaml_file, epochs=40, imgsz=640,  batch=8,  name="my_exp", exist_ok=True, augment=True,
#     fliplr=0.5,
#     flipud=0.0,
#     hsv_h=0.0,
#     hsv_s=0.0,
#     hsv_v=0.0,
#     translate=0.05,
#     scale=0.1,
#     shear=0.0,
#     perspective=0.0,
#     mosaic=0.3,
#     mixup=0.0,)

image_path = os.path.join(os.getcwd(), 'test3.mp4')
output_path = os.path.join(os.getcwd(), 'output.mp4')
# save=True, save_txt=False, save_conf=False, project='.', name='output', exist_ok=True, vid_stride=1
results = model(image_path, conf=0.1)



