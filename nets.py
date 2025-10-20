import base64
import json
import os
import re

import cv2
import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from openai import OpenAI
from PIL import Image
from scipy.special import softmax
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import create_transform
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)
from ultralytics import YOLO

import config

from pydantic import BaseModel
from typing import List


class VlAnswerSchema(BaseModel):
    description : str
    bboxs : List[List[float]]
    defects : List[str]
    


class LeafModel:
    def __init__(self):
        REPO = "rexologue/vit_large_384_for_trees"
        MODEL_NAME = "vit_large_patch16_384"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # 1) labels
        labels_path = hf_hub_download(REPO, filename="labels.json")
        with open(labels_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.labels = [raw[str(i)] for i in range(len(raw))] if isinstance(raw, dict) else list(raw)

        # 2) weights
        ckpt_path = hf_hub_download(REPO, filename="pytorch_model.bin")
        state = torch.load(ckpt_path, map_location="cpu")
        if any(k.startswith("module.") for k in state):  # DDP fix
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        # 3) model
        self.model = timm.create_model(MODEL_NAME, num_classes=len(self.labels), pretrained=False)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.DEVICE).eval()

        # 4) preprocessing (ViT-L/16 @ 384 w/ ImageNet mean/std + bicubic)
        self.transform = create_transform(
            input_size=(3, 384, 384),
            interpolation="bicubic",
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()
        topk = probs.topk(k=min(5, len(self.labels)))
        return [(self.labels[i], float(probs[i])) for i in topk.indices][0]


class YoloModel:
    def __init__(self, task, path_to_model, device="cpu"):
        self.path_to_model = path_to_model
        if task == "detect":
            model = "detect_model"
        elif task == "classify":
            model = "classify_model"
        self.task = task
        self.mask_folder = os.path.join("data", "masks")
        if not os.path.exists(self.mask_folder):
            os.makedirs(self.mask_folder, exist_ok=True)

    def predict(self, image, image_uuid):
        if self.task == "detect":
            output = self.detect(image, image_uuid)
        elif self.task == "classify":
            output = self.classify(image)
        return output

    def save_mask(self, mask, mask_filepath):
        mask_img = (mask * 255).astype(np.uint8)
        cv2.imwrite(mask_filepath, mask_img)

    def detect(self, source, source_id):
        model = YOLO(self.path_to_model)
        results = model(source, save=True)
        predictions = []
        image_w_predictions = os.path.join("data", "files", "predictions")
        os.makedirs(image_w_predictions, exist_ok=True)
        image_w_predictions = os.path.join(image_w_predictions, f"prediction_{source_id}.jpg")
        for result in results:
            result.save(image_w_predictions)

            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
            xyxy = result.boxes.xyxy
            confs = result.boxes.conf
            if result.masks is None:
                return [], image_w_predictions

            masks_data = result.masks.data

            masks_np = masks_data.cpu().numpy()
            for i, label in enumerate(names):
                bbox = [int(cord) for cord in xyxy[i]]
                mask_filepath = os.path.join(self.mask_folder, f"mask_{source_id}_{i}.png")
                self.save_mask(masks_np[i], mask_filepath)
                predictions.append(
                    {"label": label, "probability": round(float(confs[i]), 2), "bbox": bbox, "mask": mask_filepath}
                )
        return predictions, image_w_predictions

    def classify(self, image):
        model = YOLO(self.path_to_model)
        results = model(image)
        if len(results) > 2:
            return {"taxon_name_latin": "undefined", "taxon_name_ru": "undefined"}
        for r in results:
            prediction = r.names[r.probs.top1]
            probability = round(float(r.probs.top1conf), 2)
        return {
            "taxon_name_latin": prediction.replace("_", " "),
            "taxon_name_ru": self.classify_translate_taxon(prediction),
            "probability": probability,
        }

    def classify_translate_taxon(self, taxon_latin):
        with open("data/taxon_translator.json") as f:
            taxon_translator = json.load(f)
        return taxon_translator[taxon_latin.replace("_", " ")]


class DiseaseModel:
    def __init__(self):
        with open("data/defects_id2label.json") as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}
        self.processor = AutoImageProcessor.from_pretrained("OttoYu/Tree-Condition")
        self.model = AutoModelForImageClassification.from_pretrained("OttoYu/Tree-Condition")

    def predict(self, image, threshold=0.5):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits

        predictions = logits.detach().numpy()
        top_predictios = self.get_top5_predictions(model_output=predictions, id2label=self.id2label)
        return top_predictios

    def get_top5_predictions(self, model_output, id2label, use_softmax=True, threshold=0.1):
        logits = model_output[0]
        if use_softmax:
            probabilities = softmax(logits)
        else:
            probabilities = 1 / (1 + np.exp(-logits))
        indexed_probs = list(enumerate(probabilities))
        sorted_probs = sorted(indexed_probs, key=lambda x: x[1], reverse=True)
        top5 = sorted_probs[:5]
        result = {}
        for class_id, prob in top5:
            disease_name = id2label[class_id]
            if float(prob) < threshold: 
                continue
            result[disease_name] = float(prob)
        return result


class QwenModel:
    def __init__(self):
        self.prompt = """Ты умный дендролог, тебе нужно описать что не так с деревом.
                        Есть ли у него болезни, сухие ветки (если есть сухие ветки то примерно оцени их процент от дерева).
                        Если ли опасный наклон у дерева или у его веток.
                        """
                        
        self.prompt = config.VL_SYSTEM_PROMPT

    def get_description(self, image_path):
        if not os.path.exists(image_path):
            return {'description':'Error! Check HF token in config.py'}
        return self.get_output_vl(image_path)

    def resize_image_with_aspect_ratio(self, image_path, max_size=1200):
        with Image.open(image_path) as img:
            original_width, original_height = img.size

            if max(original_width, original_height) > max_size:
                aspect_ratio = original_width / original_height

                if original_width > original_height:
                    new_width = max_size
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = max_size
                    new_width = int(new_height * aspect_ratio)

                resized_img = img.resize((new_width, new_height))
                new_path = image_path.replace("images", "images_resized")
                if new_path == image_path:
                    raise ValueError(image_path, new_path)
                os.makedirs(os.path.split(new_path)[0], exist_ok=True)
                resized_img.save(new_path)
                return new_path

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def prepare_image_to_vlm(self, image_filepath):
        """
        resizes and return pseudo url of image
        """
        with Image.open(image_filepath) as image:
            width, height = image.size

        if width > config.img_max_size or height > config.img_max_size:
            path = self.resize_image_with_aspect_ratio(image_filepath, config.img_max_size)
        else:
            path = image_filepath
        base64_image = self.encode_image(path)
        pseudo_url = f"data:image/jpeg;base64,{base64_image}"
        return pseudo_url

    def get_output_vl(self, image_path):
        url = self.prepare_image_to_vlm(image_path)

        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=config.HF_TOKEN,
        )

        completion = client.beta.chat.completions.parse(
            model="Qwen/Qwen2.5-VL-72B-Instruct:nebius",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": config.VL_SYSTEM_PROMPT},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
            response_format=VlAnswerSchema,
            frequency_penalty=1,
            max_tokens=1000,
            top_p=0.9,
            # stop=[ "###", "---"],
        )

        result = completion.choices[0].message.content
        return json.loads(result)

    def get_output_vl_saved(self, image_path):
        url = self.prepare_image_to_vlm(image_path)
        has_chineese = True
        attempts_count = 0
        while has_chineese and attempts_count < 3:
            content = [{"type": "text", "text": self.prompt}]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": url},
                    ],
                },
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model.generate(**inputs, max_new_tokens=40)
            result = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1] :])
            has_chineese = re.search("[\u4e00-\u9fff]", result)

        return result
