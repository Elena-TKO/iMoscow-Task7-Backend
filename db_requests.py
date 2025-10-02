import io
import json
import os
import uuid

import cv2
import matplotlib.pyplot as plt
import numpy as np
from fastapi import HTTPException, status
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import config
from models import ImageStore, Leaf, Tree, TreeResponse, User
from nets import DiseaseModel, LeafModel, QwenModel, YoloModel

engine = create_engine(url=f"sqlite:///{config.db_path}", echo=True)
Session = sessionmaker(bind=engine)


async def analise_image(file, username="test"):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image_uuid = uuid.uuid4()
    image_path = os.path.join("data", "files", "images")
    os.makedirs(image_path, exist_ok=True)
    for_bd_path = os.path.join(image_path, f"image_{image_uuid}.png")
    image.save(for_bd_path)
    image_np = np.array(image)

    # Конвертация в BGR для OpenCV если нужно
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    with Session() as session:
        users = session.query(User).filter(User.username == username).all()
        if users == []:
            new_user = User(username="test", password="1111")
            session.add(new_user)
            session.commit()

    # Поиск деревьев на изображении (Детекция)
    detector = YoloModel("detect", path_to_model=config.detection_model)
    detection_results, image_w_predictions_path = detector.predict(image_np, image_uuid)
    # Обработка результатов детекции
    if detection_results == []:
        with Session() as session:
            user = session.query(User).filter(User.username == username).first()
            new_image = ImageStore(
                user_id=user.id,
                file_name=for_bd_path,
                processing_status=config.processing,
                preds_path_image='',
            )
            session.add(new_image)
            session.commit()
        return 100500, [], ""

    with Session() as session:
        user = session.query(User).filter(User.username == username).first()
        new_image = ImageStore(
            user_id=user.id,
            file_name=for_bd_path,
            processing_status=config.processing,
            preds_path_image=image_w_predictions_path,
        )
        session.add(new_image)
        session.flush()
        image_db_id = new_image.id

        disease_plots = []
        detected_trees_ids = []
        for i, detection in enumerate(detection_results):
            # Обрезаем изображение по bounding box
            x1, y1, x2, y2 = detection["bbox"]
            cropped_image = image_np[y1:y2, x1:x2]
            cropped_image_path = os.path.join(image_path, f"image_{image_uuid}_cropped_{i}.png")
            pil_image = Image.fromarray(cropped_image)
            pil_image.save(cropped_image_path)

            # run classifier return ['taxon_name_latin', 'taxon_name_ru']
            classifier = YoloModel("classify", path_to_model=config.classification_model)
            classification_result = classifier.predict(cropped_image, image_uuid)
            disease_model = DiseaseModel()

            disease_results = disease_model.predict(cropped_image)

            tree_image_record = Tree(
                image_id=new_image.id,
                taxon_name_latin=classification_result["taxon_name_latin"],
                taxon_name_ru=classification_result["taxon_name_ru"],
                tree_type="дерево",
                detection_probability=detection["probability"],
                classification_probability=classification_result["probability"],
                bbox_x_min=x1,
                bbox_y_min=y1,
                bbox_x_max=x2,
                bbox_y_max=y2,
                mask_file=detection["mask"],
                defects=disease_results,
                description="DEVELOPING...",
            )

            session.add(tree_image_record)
            session.flush()
            path_to_disease_plot = f"data/plots/disease_plot_{tree_image_record.id}.png"
            disease_plots.append(path_to_disease_plot)
            create_horizontal_histogram(disease_results, path_to_disease_plot)
            detected_trees_ids.append(tree_image_record.id)

        image = session.get(ImageStore, new_image.id)
        image.description = "Some image description"
        session.commit()

    trees = session.query(Tree).filter(Tree.id.in_(detected_trees_ids)).all()
    add_ids_to_image_pil(image_w_predictions_path, trees)
    return image_db_id, [TreeResponse.model_validate(tree) for tree in trees], image_w_predictions_path


async def analise_leaf(file, tree_id):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image_uuid = uuid.uuid4()
    image_path = os.path.join("data", "files", "images")
    os.makedirs(image_path, exist_ok=True)
    for_bd_path = os.path.join(image_path, f"image_leaf_{image_uuid}.png")
    image.save(for_bd_path)
    image_np = np.array(image)

    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    classifier = LeafModel()
    taxon_latin, probability = classifier.predict(for_bd_path)

    with open("data/taxons2labels.json") as f:
        t2l = json.load(f)

    with Session() as session:
        new_leaf = Leaf(
            tree_id=tree_id,
            path_to_file=for_bd_path,
            probability=probability,
            taxon_name_latin=taxon_latin,
            taxon_name_ru=t2l[taxon_latin],
        )
        session.add(new_leaf)
        session.commit()

    return {
        "tree_name_latin": taxon_latin.replace("_", " ").capitalize(),
        "tree_name_ru": t2l[taxon_latin].capitalize(),
        "classification_probability": round(probability, 2) * 100,
    }


def add_ids_to_image_pil(image_path, trees):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for tree in trees:
        bbox = [tree.bbox_x_min, tree.bbox_y_min, tree.bbox_x_max, tree.bbox_y_max]

        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        text = str(tree.id)

        bbox_height = bbox[3] - bbox[1]
        font_size = max(15, int(bbox_height * 0.10))

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

        # Размер текста
        bbox_text = draw.textbbox((0, 0), text, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]

        # Рисуем сам текст
        draw.text((center_x - text_width / 2, center_y - text_height / 2), text, fill="yellow", font=font)

    image.save(image_path)


async def get_image_with_masks(image_id):
    with Session() as session:
        image_store = session.query(ImageStore).filter(ImageStore.id == image_id).first()
        if not image_store:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

        trees = session.query(Tree).filter(Tree.image_id == image_id).all()
        # trees_info = []
        for tree in trees:
            tree_info = {  # noqa: F841
                "bbox": [tree.bbox_x_min, tree.bbox_y_min, tree.bbox_x_max, tree.bbox_y_max],
                "mask_file": tree.mask_file,
            }


async def process_image_and_mask(image_path: str, mask_path: str, bbox: list, uuid: str, id: int):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось загрузить исходное изображение")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Не удалось загрузить маску")
    if mask.shape != (img.shape[0], img.shape[1]):
        print(f"Размеры не совпадают: img={img.shape[:2]}, mask={mask.shape}. Ресайзим маску.")
        raise ValueError("Wrong image + mask size", mask.shape, img.shape)

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_img = img.copy()

    total_area = img.shape[0] * img.shape[1]
    min_area = 0.1 * total_area

    for i, contour in enumerate(contours):
        hue = np.random.randint(0, 180)
        saturation = np.random.randint(200, 256)
        value = np.random.randint(200, 256)

        hsv_color = np.uint8([[[hue, saturation, value]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        color = tuple(int(c) for c in rgb_color)

        cv2.rectangle(annotated_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

        colored_mask = np.zeros_like(img)
        colored_mask[contour_mask == 255] = color

        alpha = 0.5
        annotated_img = cv2.addWeighted(annotated_img, 1, colored_mask, alpha, 0)

    os.makedirs("data/plots", exist_ok=True)
    filename = f"image_w_mask_{uuid}_{id}.png"
    filepath = f"data/plots/{filename}"
    cv2.imwrite(filepath, annotated_img)

    return filepath


async def run_vllm(image_id):
    with Session() as session:
        image = session.get(ImageStore, image_id)
        image_path = image.file_name

        vllm_model = QwenModel()
        vllm_results = vllm_model.get_description(image_path)

        image.description = vllm_results["description"]
        session.commit()

    return True


async def get_image_description(image_id):
    with Session() as session:
        image = session.get(ImageStore, image_id)
        return image.description


async def create_horizontal_histogram(data, save_path="histogram.png"):
    """
    Создает горизонтальную гистограмму с заболеваниями по Y и вероятностью по X

    Args:
        data (dict): Словарь с данными {'disease_1': 0.3, ...}
        save_path (str): Путь для сохранения изображения
    """
    diseases = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(diseases, values, color="lightcoral", edgecolor="black", alpha=0.7, height=0.6)

    ax.set_xlabel("Вероятность", fontsize=12, fontweight="bold")
    ax.set_ylabel("Заболевания", fontsize=12, fontweight="bold")
    ax.set_title("Вероятность заболеваний", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.0)

    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.text(
            width + 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.2f}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
