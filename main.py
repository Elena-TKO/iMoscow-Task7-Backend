import base64
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

import auth
import db_requests as rq
from models import init_db
import utils as ut
import config


@asynccontextmanager
async def lifespan(app_: FastAPI):
    print("DB initialisation")
    init_db()
    print("Bot is ready")
    yield


app = FastAPI(title="To Do App", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


## --------------- PROFILE --------------- ##
@app.get("/api/user_profile/{file_id}")
async def check_connection(file_id: int):
    return rq.get_user_profile_info(file_id)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


class LoginRequest(BaseModel):
    login: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict


@app.post("/api/auth/login", response_model=Token)
async def login(login_data: LoginRequest):
    # Ищем пользователя
    user = auth.get_user_by_login(login_data.login)

    if not user or not auth.verify_password(login_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверные учетные данные",
        )

    # Создаем токен
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)

    user_response = {
        "id": user.id,
        "username": user.username,
        "icon": user.user_icon,
        "full_name": user.full_name,
    }

    return {"access_token": access_token, "token_type": "bearer", "user": user_response}


@app.post("/api/analyze-leaf/{tree_id}")
async def analyze_image(tree_id: int, image: UploadFile):
    """
    Эндпоинт для анализа изображения с деревьями
    """

    return await rq.analise_leaf(image, tree_id)


@app.post("/api/analyze-image")
async def analyze_image(
    background_tasks: BackgroundTasks,
    image: UploadFile,
):
    """
    Эндпоинт для анализа изображения с деревьями
    """
    validation = ut.validate_image_file(image)
    if validation != True:
        return config.https_errors[validation]

    image_db_id, response_data, image_w_predictions_path = await rq.analise_image(image)

    background_tasks.add_task(
        rq.run_vllm,
        image_db_id,
    )
    if image_w_predictions_path != "":
        with open(image_w_predictions_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
    else:
        image_base64 = ""
    return {
        "trees": response_data,
        "tree_table_data": tree_to_table(response_data),
        "image_w_prediction": image_base64,
        "image_id": image_db_id,
    }


def tree_to_table(trees):
    accepted_columns = [
        "id",
        "taxon_name_latin",
        "taxon_name_ru",
        "tree_type",
        "classification_probability",
        "defects",
        "description",
    ]
    table_data = []

    for tree in trees:
        prob = round(float(tree.classification_probability), 2)
        prob = prob * 100
        table_data.append(
            {
                "ID": tree.id,
                "Название L": tree.taxon_name_latin,
                "Названия RU": tree.taxon_name_ru,
                "Тип": tree.tree_type,
                "Вероятность": prob,
                "Дефекты": "\n".join([f"{k}: {round(v,2)*100}" for k, v in tree.defects.items()]),
                "Описание": tree.description,
            }
        )
    return table_data


@app.get("/api/processing-status/{image_id}")
async def get_processing_status(image_id: str):
    """Эндпоинт для проверки статуса обработки"""
    vlm_description = await rq.get_image_description(image_id)
    if vlm_description == config.processing:
        return {"status": "processing", "description": "generating...", 'vlm_bbox_defect_mask':''}
    else:
        return {"status": "completed", "description": vlm_description, }
