import os
import signal
from contextlib import contextmanager

from PIL import Image

from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE
import cv2
from math import atan2, cos, sin, sqrt, pi
import numpy as np
import matplotlib.pyplot as plt

class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Время выполнения истекло")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def resize_image(input_path, new_width=None, new_height=None):
    """
    Изменяет размер изображения с сохранением пропорций.

    :param input_path: Путь к исходному изображению
    :param output_path: Путь для сохранения результата
    :param new_width: Новая ширина (если None, рассчитывается по высоте)
    :param new_height: Новая высота (если None, рассчитывается по ширине)
    """
    input_path = os.path.join(os.getcwd(), input_path)
    output_path = os.path.join(os.path.dirname(input_path), "resized")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, os.path.basename(input_path))
    with Image.open(input_path) as img:
        width, height = img.size

        # Если указана только одна сторона, вычисляем вторую с сохранением пропорций
        if new_width is None and new_height is not None:
            ratio = new_height / height
            new_width = int(width * ratio)
        elif new_height is None and new_width is not None:
            ratio = new_width / width
            new_height = int(height * ratio)
        elif new_width is None and new_height is None:
            raise ValueError("Нужно указать new_width или new_height")

        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        resized_img.save(output_path)
        print(f"Изображение сохранено: {output_path} ({new_width}x{new_height})")
    return output_path


def validate_image_file(file) -> None:
    """Валидация загружаемого изображения"""
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        return "wrong_file_extention"

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_FILE_SIZE:
        return "large_file"

    return True


# -- image processing --

def drawAxis(img, p_, q_, color=(255, 255, 255), scale=1.0):
    p = list(p_)
    q = list(q_)

    # Вычисляем угол и длину
    angle = atan2(p[1] - q[1], p[0] - q[0])  # угол в радианах
    hypotenuse = sqrt((p[1] - q[1]) ** 2 + (p[0] - q[0]) ** 2)

    # Удлиняем стрелку
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)

    # Основная линия
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)

    # "Крючки" стрелки
    hook_len = 7  # немного меньше, чтобы выглядело аккуратнее
    p[0] = q[0] + hook_len * cos(angle + pi / 4)
    p[1] = q[1] + hook_len * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)

    p[0] = q[0] + hook_len * cos(angle - pi / 4)
    p[1] = q[1] + hook_len * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)

def getOrientation(pts, img):
  ## [pca]
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
 
  # Store the center of the object
  cntr = (int(mean[0,0]), int(mean[0,1]))
  ## [pca]
 
  ## [visualization]
  # Draw the principal components
  cv2.circle(img, cntr, 3, (255, 0, 255), 2)
  p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
  drawAxis(img, cntr, p1, (255, 255, 0), 1)
 
  angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
  ## [visualization]
 
  # Label with the rotation angle
  label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) + 90) + " degrees"
  cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
 
  return angle
 
 
def addAxis(image_path_w_mask, mask_path):
    image = cv2.imread(image_path_w_mask, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    areas = [cv2.contourArea(c) for c in contours]
    max_index = areas.index(max(areas))

    # сам контур
    largest_contour = contours[max_index]
    # Draw each contour only for visualisation purposes
    # cv2.drawContours(mask_color, [largest_contour], 0, (0, 100, 255), 2)
    # Find the orientation of each shape
    getOrientation(largest_contour, image)
    
    cv2.imwrite(image_path_w_mask, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    


from PIL import Image, ImageDraw, ImageFont
import os

def save_bbox_mask(image_path, bboxs, defects, output_dir="data/files/masks"):
    """
    Создает или обновляет PNG-маску с прозрачным фоном, добавляя bbox'ы и подписи.
    Если файл уже существует — открывает его и дорисовывает новые рамки.
    bboxs — список списков [x_min, y_min, w, h] (YOLO-нормализованные координаты)
    defects — список строк (названия дефектов)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Имя файла без индексов
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")

    # Проверяем, есть ли уже маска
    if os.path.exists(mask_path):
        # Открываем существующую маску
        mask = Image.open(mask_path).convert("RGBA")
    else:
        # Узнаем размер исходного изображения
        base_image = Image.open(image_path).convert("RGBA")
        width, height = base_image.size
        # Создаем новую прозрачную маску
        mask = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    draw = ImageDraw.Draw(mask)

    # Загружаем шрифт (arial или стандартный)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()

    width, height = mask.size

    # Отрисовываем все bbox и подписи
    for bbox, defect in zip(bboxs, defects):
        x_min, y_min, w, h = bbox
        x1 = int(x_min * width)
        y1 = int(y_min * height)
        x2 = int((x_min + w) * width)
        y2 = int((y_min + h) * height)

        # Рамка bbox (белая)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255, 255), width=4)

        # Подпись
        text_bg = (255, 255, 255, 180)
        text_bbox = draw.textbbox((x1, y1), defect, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 6, y1], fill=text_bg)
        draw.text((x1 + 3, y1 - text_h - 2), defect, fill=(0, 0, 0, 255), font=font)

    # Сохраняем (перезаписывая тем же именем, но с добавленными объектами)
    mask.save(mask_path, "PNG")

    return mask_path
