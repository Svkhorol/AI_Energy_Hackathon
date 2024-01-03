from imgsearch.iou import compute_iou
from imgsearch import config
from bs4 import BeautifulSoup
from imutils import paths
import cv2
import os

# Режим мультипроцессорности
cv2.setUseOptimized(True)
cv2.setNumThreads(8)

# Список с путями ко всем train-изображениям
imagePaths = list(paths.list_images(config.ORIG_IMAGES))

totalPositive = 0
totalNegative = 0
totalGT = 0

for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}...".format(i + 1, len(imagePaths)))
    # Извлекаем имя фото
    filename = imagePath.split(os.path.sep)[-1]
    filename = filename[:filename.rfind(".")]
    annotPath = os.path.sep.join([config.ORIG_IMAGES,
                                  "{}.xml".format(filename)])
    # Загружаем файл с аннотацией, инициализируем список с GT-регионами
    contents = open(annotPath).read()
    soup = BeautifulSoup(contents, "html.parser")
    gtBoxes = []
    # Размер фото
    w = int(soup.find("width").string)
    h = int(soup.find("height").string)

    # Идём по каждому боксу в xml-файле (тег <object>)
    for o in soup.find_all("object"):
        # Извлекаем метку <name> и координаты
        label = o.find("name").string
        xMin = int(o.find("xmin").string)
        yMin = int(o.find("ymin").string)
        xMax = int(o.find("xmax").string)
        yMax = int(o.find("ymax").string)
        # Границы бокса не должны попадать за границы фото
        xMin = max(0, xMin)
        yMin = max(0, yMin)
        xMax = min(w, xMax)
        yMax = min(h, yMax)

        gtBoxes.append((xMin, yMin, xMax, yMax))

    image = cv2.imread(imagePath)

    # Сохранение фото для реальных боксов
    for gtBox in gtBoxes:
        # ground-truth box
        (gtStartX, gtStartY, gtEndX, gtEndY) = gtBox
        roi = None
        roi = image[gtStartY:gtEndY, gtStartX:gtEndX]
        filename = "GTBox_{}.png".format(totalGT)
        outputPath = os.path.sep.join([config.POSITVE_PATH, filename])
        totalGT += 1
        if roi is not None and outputPath is not None:
            roi = cv2.resize(roi, config.INPUT_DIMS,  # для нейросети
                             interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(outputPath, roi)

    # Селективный поиск по фото
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    proposedRects = []

    for (x, y, w, h) in rects:
        # создаем прямоугольник
        proposedRects.append((x, y, x + w, y + h))

    # Какие регионы лучше всего совпадают с реальными боксами по IoU
    positiveROIs = 0
    negativeROIs = 0

    for proposedRect in proposedRects[:config.MAX_PROPOSALS]:
        # proposed rectangle
        (propStartX, propStartY, propEndX, propEndY) = proposedRect
        for gtBox in gtBoxes:
            iou = compute_iou(gtBox, proposedRect)

            # ground-truth box
            (gtStartX, gtStartY, gtEndX, gtEndY) = gtBox
            # initialize ROI for proposal and output path
            roi = None
            outputPath = None
            # Если IoU > 70%
            if iou > 0.7 and positiveROIs <= config.MAX_POSITIVE:
                # извлекаем ROI и отправляем в папку с положительным классом
                roi = image[propStartY:propEndY, propStartX:propEndX]
                filename = "{}.png".format(totalPositive)
                outputPath = os.path.sep.join([config.POSITVE_PATH, filename])
                positiveROIs += 1
                totalPositive += 1
            # Если IoU < 5%
            if iou < 0.05 and negativeROIs <= config.MAX_NEGATIVE:
                # извлекаем ROI и отправляем в папку с отрицательным классом
                roi = image[propStartY:propEndY, propStartX:propEndX]
                filename = "{}.png".format(totalNegative)
                outputPath = os.path.sep.join([config.NEGATIVE_PATH, filename])
                negativeROIs += 1
                totalNegative += 1
            # Отправляем все регионы в датасет
            if roi is not None and outputPath is not None:
                roi = cv2.resize(roi, config.INPUT_DIMS,    # для нейросети
                                 interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(outputPath, roi)
