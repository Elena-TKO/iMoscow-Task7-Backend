from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import config
from models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

engine = create_engine(url=f"sqlite:///{config.db_path}", echo=True)
Session = sessionmaker(bind=engine)

SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Хэширование паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class LoginRequest(BaseModel):
    login: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict


class UserResponse(BaseModel):
    id: int
    user_icon: str
    username: str
    full_name: str


class UserCreateRequest(BaseModel):
    username: str
    password: str
    full_name: str
    user_icon: str


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user_by_login(login: str):
    """Ищем пользователя username"""
    with Session() as session:
        user = session.query(User).filter((User.username == login)).first()
        return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверные учетные данные",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user_by_login(username)
    if user is None:
        raise credentials_exception
    return user


def add_new_user(user_data: UserCreateRequest):
    hashed_password = get_password_hash(user_data.password)

    with Session() as session:
        existing_user = session.query(User).filter(User.username == user_data.username).first()
        if existing_user:
            return
        new_user = User(
            username=user_data.username,
            password=hashed_password,
            user_icon=user_data.user_icon,
            full_name=user_data.full_name,
        )
        session.add(new_user)
        session.commit()
        session.refresh(new_user)
    return new_user
