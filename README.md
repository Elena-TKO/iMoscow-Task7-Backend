### Бэкенд PhytoScan

Этот репозиторий содержит бэкенд для PhytoScan.
Скорость инфреренса пайплайна всех моделей как на gpu, так и на cpu составляет до 30 секунд.

**ВНИМАНИЕ!** 
*Однако при первом запуске будет происходить скачивание моделей с HF, что может занять 1-10 минут
в зависимости от вашей скорости интернета!*

## Установка
Версия питона: python=3.11

Установить pytorch https://pytorch.org/get-started/locally/

Установка библиотек:
```
pip install python-multipart
pip install timm
pip install uvicorn
pip install -r requirements.txt
```

## Скачивание весов моделей

Модели детекции и классфикации доступны по ссылку Дополнительно в 
приложенном решении нашей команды.
Скачайте папку `models` и поместите ее в кореневую директорию проекта.

Создайте HF токен и пропишите в файл `config.py` :
```
HF_TOKEN = "<ваш токен>"
```

Заменить `http://localhost:9000` на `ваш порт фронтенда`
```
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
## Запуск

```
uvicorn main:app --reload
```

## Code style

```
black --safe --line-length=120 .
isort --profile black .
```
или 
```
make lint
```

