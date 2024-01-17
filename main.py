import os
from imgsearch import config
from imutils import paths
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO


yolov8_path = 'models/yolo_best.pt'
model = YOLO(yolov8_path)

imagePaths = list(paths.list_images(config.TEST_PATH))
RESULT_PATH = 'imgs_pred/'


detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_path,
    device="cpu",
)

# Инференс на тестовых изображениях
for (i, imagePath) in enumerate(imagePaths):

    # Название фото
    filename = imagePath.split(os.path.sep)[-1]
    filename = filename[:filename.rfind(".")]
    print("[INFO] processing image {}/{}...".format(i + 1, len(imagePaths)))

    result = get_sliced_prediction(
        imagePath,
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # Сохранение изображений с предсказаниями
    result.export_visuals(
        export_dir=RESULT_PATH, file_name=filename
    )
