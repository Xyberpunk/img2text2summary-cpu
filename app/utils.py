import os, uuid, pathlib, shutil
from typing import Tuple

STORAGE_DIR = os.getenv("STORAGE_DIR", "/data")
pathlib.Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)

def save_upload_to_storage(fileobj, filename: str) -> str:
    ext = filename.split('.')[-1].lower()
    job_id = str(uuid.uuid4())
    job_dir = pathlib.Path(STORAGE_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    target = job_dir / f"input.{ext}"
    with open(target, "wb") as f:
        shutil.copyfileobj(fileobj, f)
    return str(target), job_id
