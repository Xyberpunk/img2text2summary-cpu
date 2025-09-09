from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class JobCreate(BaseModel):
    pass  # empty; file handled separately

class JobRead(BaseModel):
    id: str
    status: str
    source: Optional[str]
    ocr_chars: int
    ocr_conf: float
    summary: Optional[str]
    error: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
