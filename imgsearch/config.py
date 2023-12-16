import os

# Путь к исходным изображениям для трейна
ORIG_BASE_PATH = "train"
# ORIG_IMAGES = "train"
# ORIG_ANNOTS = "train"

# Путь к созданному датасету после build_dataset.py
BASE_PATH = "dataset"
POSITVE_PATH = os.path.sep.join([BASE_PATH, "defect"])
NEGATIVE_PATH = os.path.sep.join([BASE_PATH, "no_defect"])


# Количество регионов Selective Search
MAX_PROPOSALS = 2000         # train
MAX_PROPOSALS_INFER = 200    # predict

# Максимальное количество регионов каждого класса
MAX_POSITIVE = 30
MAX_NEGATIVE = 10

# Параметры ml-модели
INPUT_DIMS = (224, 224)
MODEL_PATH = "defect_detector.h5"
ENCODER_PATH = "label_encoder.pickle"
MIN_PROBA = 0.90
