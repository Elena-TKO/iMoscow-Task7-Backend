import os
import signal
from contextlib import contextmanager

from PIL import Image

from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE


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
