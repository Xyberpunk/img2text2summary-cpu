import os, cv2, numpy as np
from statistics import mean
from app.celery_app import celery_app

# Initialize PaddleOCR once (download weights on first run)
from paddleocr import PaddleOCR

OCR_LANG = os.getenv("OCR_LANG", "en")
OCR_VERSION = os.getenv("OCR_VERSION", "PP-OCRv5")

ocr = PaddleOCR(
    use_angle_cls=True,
    lang=OCR_LANG,
    ocr_version=OCR_VERSION,
    use_gpu=False
)

def preprocess(path: str) -> str:
    # Simple CPU-friendly cleanup: grayscale -> denoise -> threshold -> deskew
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # Otsu binarization
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Deskew using moments
    coords = np.column_stack(np.where(th < 255))
    angle = 0.0
    if coords.size > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
    (h, w) = th.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(th, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    tmp = path + ".pre.png"
    cv2.imwrite(tmp, rotated)
    return tmp

@celery_app.task(name="ocr.run")
def ocr_run(image_path: str):
    prepped = preprocess(image_path)
    result = ocr.ocr(prepped, cls=True)
    texts = []
    confs = []
    # PaddleOCR returns list per image; each with [ [box, (text, conf)], ... ]
    if result and result[0]:
        for block in result[0]:
            txt, cf = block[1]
            texts.append(txt)
            confs.append(float(cf))
    text_full = " ".join(texts).strip()
    mean_conf = float(sum(confs)/len(confs)) if confs else 0.0
    return {"text": text_full, "mean_conf": mean_conf, "n_boxes": len(texts)}
