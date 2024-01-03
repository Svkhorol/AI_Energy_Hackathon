import os

# Пути к исходным изображениям для Selective Search
ORIG_BASE_PATH = "train"
ORIG_IMAGES = "train"

# Путь к изображениям для YOLO
BASE_PATH = "datasets"
DATASET_PATH = os.path.sep.join([BASE_PATH, "data.yaml"])

# Пути к изображениям для теста
TEST_PATH = "test"
VALID_PATH = "valid"

# Пути к созданному датасету после build_dataset
POSITVE_PATH = os.path.sep.join([BASE_PATH, "defect"])
NEGATIVE_PATH = os.path.sep.join([BASE_PATH, "no_defect"])

# Количество регионов Selective Search
MAX_PROPOSALS = 2000         # train
MAX_PROPOSALS_INFER = 1000    # predict

# Максимальное количество регионов каждого класса
MAX_POSITIVE = 30
MAX_NEGATIVE = 10

# Параметры resnet-модели
INPUT_DIMS = (224, 224)
MODEL_PATH = "models/defect_detector.h5"
ENCODER_PATH = "models/label_defect.pickle"
MIN_PROBA = 0.80
