from fastapi import HTTPException, status

creds_path = "creds.json"
promt_folder_path = "data/promts"

# DATABASE
db_path = "db.sqlite3"

processing = "processing..."
# DETECTORS
detection_model = "models/best_detect.pt"
classification_model = "models/best_classify.pt"
# LLM
HF_TOKEN = "<ваш токен>"

VL_SYSTEM_PROMPT = """Ты — опытный дендролог-эксперт с 20-летним стажем работы. Твоя задача — проводить комплексный анализ деревьев по фотографиям и предоставлять краткое профессиональное заключение.
Оцени такие параметры:
наклон дерева (опасный, небольшой, отсутствует)), заболевания и повреждения, наличие сухих ветвей или сухостоя
Если какой-то параметр в норме то не указывай его. Не пиши недостоверную информацию.
Анализируй предоставленное изображение дерева и составь небольшое описание 2-3 предложения. Не пиши много текста, и отвечай только если уверен в своем анализе.
Если на дереве нет листьев (фото сделано осенью/зимой), то не пиши про сухие ветки.
"""
extention_to_mimetype = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/msword",
    ".md": "text/markdown",
}

ALLOWED_EXTENSIONS = {"jpeg", "jpg", "png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 МБ
img_max_size = 300

https_errors = {
    "wrong_file_extention": HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Размер файла превышает лимит {MAX_FILE_SIZE // 1024 // 1024} МБ.",
    ),
    "large_file": HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST, detail="Неподдерживаемый формат файла. Разрешены только JPEG и PNG."
    ),
}
