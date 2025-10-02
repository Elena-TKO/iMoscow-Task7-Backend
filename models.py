from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
)

import auth
import config

engine = create_engine(url=f"sqlite:///{config.db_path}", echo=True)
Session = sessionmaker(bind=engine)


class Base(AsyncAttrs, DeclarativeBase):
    pass


def get_utc_now():
    """Возвращает текущее время в UTC с timezone awareness"""
    return datetime.now(timezone.utc)


class TreeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_id: int
    taxon_name_latin: str
    taxon_name_ru: str
    tree_type: str
    detection_probability: float
    classification_probability: float
    bbox_x_min: float
    bbox_y_min: float
    bbox_x_max: float
    bbox_y_max: float
    mask_file: Optional[str] = None
    defects: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


## -------------------- COMMON -------------------------
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username = Column(String)
    user_icon = Column(String)
    password = Column(String)
    full_name = Column(String)


class Tree(Base):
    __tablename__ = "trees"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(Integer, ForeignKey("image_stores.id"))

    taxon_name_latin: Mapped[str] = mapped_column(String(100))
    taxon_name_ru: Mapped[str] = mapped_column(String(100))
    tree_type: Mapped[str] = mapped_column(String(50))
    detection_probability = mapped_column(Float)
    classification_probability = mapped_column(Float)
    # percent_of_dry_branches: Mapped[float] = mapped_column()  # Процент сухих ветвей

    bbox_x_min: Mapped[float] = mapped_column(Float)
    bbox_y_min: Mapped[float] = mapped_column(Float)
    bbox_x_max: Mapped[float] = mapped_column(Float)
    bbox_y_max: Mapped[float] = mapped_column(Float)

    mask_file: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    defects: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # image: Mapped["ImageStore"] = relationship("ImageStore", back_populates="trees")
    # leaves: Mapped[List["Leaf"]] = relationship("Leaf", back_populates="tree", cascade="all, delete-orphan")


class ImageStore(Base):
    __tablename__ = "image_stores"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"))
    file_name: Mapped[str] = mapped_column(String(255))
    processing_status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=get_utc_now)
    preds_path_image: Mapped[str] = mapped_column(String(255))
    # processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String(255))

    # user: Mapped["User"] = relationship("User", back_populates="image_stores")


class Leaf(Base):
    __tablename__ = "leafs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tree_id: Mapped[int] = mapped_column(Integer, ForeignKey("trees.id"))
    path_to_file = mapped_column(String(255))
    probability = mapped_column(String(255))
    taxon_name_latin: Mapped[str] = mapped_column(String(100))  # Вид дерева (дуб, клен, береза и т.д.)
    taxon_name_ru: Mapped[str] = mapped_column(String(100))

    # tree: Mapped["Tree"] = relationship("Tree", back_populates="leaves")


def init_db():
    """Создает все таблицы в базе данных"""
    Base.metadata.create_all(bind=engine)

    user_data = auth.UserCreateRequest(
        **{"username": "admin", "user_icon": "path_to_icon", "password": "1111", "full_name": "I AM ADMIN"}
    )
    auth.add_new_user(user_data)
