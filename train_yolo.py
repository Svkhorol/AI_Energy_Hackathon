import os
import torch
from imgsearch import config
from ultralytics import settings
from ultralytics import YOLO

# GPU
# torch.cuda.set_device(0)

# Update ultralytics setting
settings.update({'datasets_dir': os.getcwd()})

# Размер изображений для обучения
# 1280
# 3008
INPUT_SIZE = 960

# Build a new model
model = YOLO('yolov8n.yaml')

results = model.train(
    data=config.DATASET_PATH,
    imgsz=INPUT_SIZE,
    epochs=100,
    close_mosaic=10,
    # device=0,
    plots=True
)
