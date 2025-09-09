from sqlmodel import SQLModel, Field, Session, create_engine, select
from typing import Optional
from datetime import datetime
import os, uuid, pathlib

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/app.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})

class Job(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    status: str = "queued"        # queued, processing, done, error
    source: Optional[str] = None  # 'ocr' or 'caption'
    image_path: str
    ocr_chars: int = 0
    ocr_conf: float = 0.0
    summary: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    return Session(engine)
