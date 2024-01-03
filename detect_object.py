from imgsearch import config
from imutils import paths
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os
import pandas as pd

imagePaths = list(paths.list_images(config.TEST_PATH))

print("[INFO] loading model and label binarizer...")
model = load_model(config.MODEL_PATH)
lb = pickle.loads(open(config.ENCODER_PATH, "rb").read())

# Пустой DataFrame для записи сабмита
COLs = ['file_name', 'rbbox', 'probability']
submit_df = pd.DataFrame(columns=COLs)

for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}...".format(i + 1, len(imagePaths)))

    # Извлекаем название фото
    filename = imagePath.split(os.path.sep)[-1]
    # Загрузка фото для инференса
    image = cv2.imread(imagePath)

    # Селективный поиск регионов
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()

    proposals = []
    boxes = []
    for (x, y, w, h) in rects[:config.MAX_PROPOSALS_INFER]:
        # Обработка каждого региона для подачи в модель
        roi = image[y:y + h, x:x + w]
        # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, config.INPUT_DIMS,
                         interpolation=cv2.INTER_CUBIC)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)

        # Добавляем регионы и их координаты в список
        proposals.append(roi)
        boxes.append((x, y, x + w, y + h))

    # Преобразование списка регионов в массив
    proposals = np.array(proposals, dtype="float32")
    boxes = np.array(boxes, dtype="int32")

    # Классификация регионов моделью ResNet
    proba = model.predict(proposals)

    # Отбираем положительный класс с дефектом
    labels = lb.classes_[np.argmax(proba, axis=1)]
    idxs = np.where(labels == "defect")[0]
    # Отбираем соотвествующие классу боксы
    boxes = boxes[idxs]
    proba = proba[idxs][:, 1]
    # Отбираем боксы с вероятностью от 80%
    idxs = np.where(proba >= config.MIN_PROBA)
    boxes = boxes[idxs]
    proba = proba[idxs]
    print(boxes)

    # Временный df для одного изображения
    temp_df = pd.DataFrame(columns=COLs)
    temp_df.loc[0, 'file_name'] = filename[:-4]

    # Извлекаем координаты углов bounding box'ов
    x1 = boxes[:, 0]  # x-координата верхнего левого угла
    y1 = boxes[:, 1]  # y-координата верхнего левого угла
    x2 = boxes[:, 2]  # x-координата нижнего правого угла
    y2 = boxes[:, 3]  # y-координата нижнего правого угла

    # x, y - центр прямоугольника в относительных координатах
    # w, h - ширина и высота в относительных координатах
    x = (x1 + x2) / 2 / image.shape[1]
    y = (y1 + y2) / 2 / image.shape[0]
    w = (x2 - x1) / image.shape[1]
    h = (y2 - y1) / image.shape[0]

    boxes = np.column_stack([x, y, w, h])

    temp_df['rbbox'] = str(boxes.tolist())
    temp_df['probability'] = str(proba.tolist())

    # Добавляем временный df к основному результату
    submit_df = pd.concat([submit_df, temp_df], ignore_index=True)

    # Построчная запись в CSV-файл
    submit_df.to_csv('result.csv', index=False)
    print("[INFO] Results saved to csv-file")

# Визуализация результата
