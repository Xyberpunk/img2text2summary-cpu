# app/main.py
"""
Image → Text → Summary (CPU-only)
- Single-file FastAPI app (no imports from app.infer)
- PaddleOCR (CPU) for OCR with light preprocessing (deskew/denoise/contrast)
- HuggingFace summarization (CPU) with chunking + per-request token controls
- Bullet / Markdown formatting w/ adjustable number of points
- CORS enabled for Vercel or local UIs
"""

# --- Keep torchvision out for CPU-only installs (prevents torchvision import paths) ---
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# =============================
# Config (env + sane defaults)
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "data")))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# OCR config (PaddleOCR 2.7.x supports up to v4)
_OCR_ALLOWED = {"PP-OCR", "PP-OCRv2", "PP-OCRv3", "PP-OCRv4"}
OCR_VERSION_RAW = os.getenv("OCR_VERSION", "PP-OCRv4")
OCR_LANG = os.getenv("OCR_LANG", "en")
OCR_MIN_CONF = float(os.getenv("OCR_MIN_CONF", "0.60"))

# Preprocess toggles
PRE_DENOISE = os.getenv("PRE_DENOISE", "1") == "1"
PRE_DESKEW = os.getenv("PRE_DESKEW", "1") == "1"
PRE_CONTRAST = os.getenv("PRE_CONTRAST", "1") == "1"

# Summarization defaults (can be overridden per request via query params)
SUMMARY_MODEL_ID = os.getenv("SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")
SUMMARY_MAX_NEW_TOKENS = int(os.getenv("SUMMARY_MAX_NEW_TOKENS", "140"))
SUMMARY_MIN_NEW_TOKENS = int(os.getenv("SUMMARY_MIN_NEW_TOKENS", "40"))
SUMMARY_NUM_BEAMS = int(os.getenv("SUMMARY_NUM_BEAMS", "5"))
SUMMARY_LENGTH_PENALTY = float(os.getenv("SUMMARY_LENGTH_PENALTY", "1.1"))
SUMMARY_NGRAM_BLOCK = int(os.getenv("SUMMARY_NGRAM_BLOCK", "3"))

# =============================
# Logging
# =============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("img2text2summary")

def _normalize_ocr_version(v: str) -> str:
    v_up = (v or "").strip().upper()
    canon = {
        "PP-OCR": "PP-OCR",
        "PP-OCRV2": "PP-OCRv2",
        "PP-OCRV3": "PP-OCRv3",
        "PP-OCRV4": "PP-OCRv4",
    }
    if v_up in canon:
        return canon[v_up]
    if v_up == "PP-OCRV5":
        log.warning("OCR_VERSION=PP-OCRv5 not supported by paddleocr==2.7.x; using PP-OCRv4.")
        return "PP-OCRv4"
    if v and v not in _OCR_ALLOWED:
        log.warning(f"OCR_VERSION={v!r} not supported; using PP-OCRv4.")
    return "PP-OCRv4"

OCR_VERSION = _normalize_ocr_version(OCR_VERSION_RAW)

# =============================
# Lazy backends
# =============================
_ocr_backend = None                 # PaddleOCR instance
_hf_pipeline = None                 # transformers pipeline
_summarizer_backend: Optional[str] = None  # "hf" | "simple" | None

def summarizer_backend_name() -> str:
    return _summarizer_backend or "init_on_first_use"

# =============================
# OCR (PaddleOCR + preprocessing)
# =============================
def get_ocr_backend():
    """Initialize PaddleOCR once (CPU)."""
    global _ocr_backend
    if _ocr_backend is None:
        from paddleocr import PaddleOCR  # heavy import done lazily
        log.info(f"Initializing PaddleOCR(lang={OCR_LANG}, version={OCR_VERSION}) on CPU…")
        _ocr_backend = PaddleOCR(
            use_angle_cls=True,
            lang=OCR_LANG,
            ocr_version=OCR_VERSION,
            use_gpu=False,
            show_log=False,
        )
        log.info("PaddleOCR ready.")
    return _ocr_backend

def _read_image_from_upload(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return img

def _preprocess(img: np.ndarray) -> np.ndarray:
    """Light CPU-friendly preprocessing to help OCR."""
    out = img.copy()

    if PRE_DESKEW:
        try:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thr == 0))  # black pixels
            if coords.size > 0:
                angle = cv2.minAreaRect(coords)[-1]
                angle = -(90 + angle) if angle < -45 else -angle
                (h, w) = gray.shape
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except Exception as e:
            log.debug(f"Deskew skipped: {e}")

    if PRE_DENOISE:
        try:
            out = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)
        except Exception as e:
            log.debug(f"Denoise skipped: {e}")

    if PRE_CONTRAST:
        try:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            out = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception as e:
            log.debug(f"Contrast skipped: {e}")

    return out

def run_ocr(img: np.ndarray) -> Tuple[str, float, int]:
    """
    OCR with reading-order sort (top→bottom, then left→right).
    Returns: (joined_text, avg_conf, char_count)
    """
    ocr = get_ocr_backend()
    pre = _preprocess(img)
    result = ocr.ocr(pre, cls=True)

    lines: List[str] = []
    confs: List[float] = []

    for page in result:
        # Each page item: [points, (text, conf)]
        page_sorted = sorted(
            page,
            key=lambda item: (
                min(p[1] for p in item[0]),  # top (y)
                min(p[0] for p in item[0]),  # left (x)
            ),
        )
        for item in page_sorted:
            text = item[1][0].strip()
            conf = float(item[1][1])
            if text and conf >= OCR_MIN_CONF:
                lines.append(text)
                confs.append(conf)

    joined = "\n".join(lines).strip()
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return joined, avg_conf, len(joined)

# =============================
# Summarization (HF + fallback)
# =============================
_STOPWORDS = set("""
a an and are as at be by for from has have he her hers him his i in is it its of on or our she that the their them they this to was we were will with you your
""".strip().split())
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def simple_extractive_summarize(text: str, max_sentences: int = 3) -> str:
    """Tiny extractive: rank sentences by keyword frequency + small lead bias."""
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if not sents:
        return text.strip()

    freqs = {}
    for w in re.findall(r"[A-Za-z0-9']+", text.lower()):
        if w in _STOPWORDS:
            continue
        freqs[w] = freqs.get(w, 0) + 1
    if not freqs:
        return " ".join(sents[:max_sentences])

    scored = []
    for idx, s in enumerate(sents):
        sw = re.findall(r"[A-Za-z0-9']+", s.lower())
        score = sum(freqs.get(w, 0) for w in sw) / (len(sw) + 1e-6)
        score += max(0.0, 1.0 - (idx / max(1, len(sents))))  # small lead bias
        scored.append((score, idx, s))

    top = sorted(scored, key=lambda x: (-x[0], x[1]))[:max_sentences]
    top_sorted = [s for (_, _, s) in sorted(top, key=lambda x: x[1])]
    return " ".join(top_sorted)

def try_init_hf_summarizer() -> bool:
    """Initialize a Hugging Face summarizer (CPU)."""
    global _hf_pipeline, _summarizer_backend
    if _hf_pipeline is not None:
        return True
    if _summarizer_backend == "simple":
        return False
    try:
        from transformers import pipeline  # type: ignore
        log.info(f"Initializing HF summarizer: {SUMMARY_MODEL_ID} (CPU)")
        _hf_pipeline = pipeline(
            "summarization",
            model=SUMMARY_MODEL_ID,
            tokenizer=SUMMARY_MODEL_ID,
            device=-1,       # CPU
            framework="pt",  # requires torch installed
        )
        _summarizer_backend = "hf"
        log.info("HF summarizer ready.")
        return True
    except Exception as e:
        log.warning(f"HF summarizer not available ({e}). Using simple extractive.")
        _summarizer_backend = "simple"
        return False

def _hf_tokenizer():
    """Tokenizer compatible with the current HF pipeline/model."""
    try:
        return _hf_pipeline.tokenizer  # type: ignore[attr-defined]
    except Exception:
        from transformers import AutoTokenizer  # type: ignore
        return AutoTokenizer.from_pretrained(SUMMARY_MODEL_ID)

def _chunks_by_tokens(text: str, max_tokens: int = 800, overlap: int = 120):
    """Split long text into token windows with overlap so we don't lose the tail."""
    tok = _hf_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    n = len(ids)
    if n <= max_tokens:
        yield text
        return
    i = 0
    while i < n:
        window = ids[i:i + max_tokens]
        yield tok.decode(window, skip_special_tokens=True)
        if i + max_tokens >= n:
            break
        i += max_tokens - overlap

def _summarize_chunked(text: str, mn: Optional[int], mx: Optional[int]) -> str:
    """
    Map-reduce style summary over token chunks, then a final pass.
    mn/mx: per-request min/max new tokens (HF only).
    """
    assert _hf_pipeline is not None

    # Base from env, allow per-request overrides
    base_mx = max(8, SUMMARY_MAX_NEW_TOKENS)
    base_mn = min(SUMMARY_MIN_NEW_TOKENS, base_mx // 2)
    use_mx = int(mx) if isinstance(mx, int) else base_mx
    use_mn = int(mn) if isinstance(mn, int) else base_mn
    use_mn = max(0, min(use_mn, use_mx // 2))

    partials: List[str] = []
    for piece in _chunks_by_tokens(text, max_tokens=800, overlap=120):
        out = _hf_pipeline(
            piece,
            max_new_tokens=min(use_mx, 120),
            min_new_tokens=min(use_mn, 60),
            num_beams=SUMMARY_NUM_BEAMS,
            length_penalty=SUMMARY_LENGTH_PENALTY,
            no_repeat_ngram_size=SUMMARY_NGRAM_BLOCK,
        )
        partials.append((out[0].get("summary_text") or "").strip())

    combined = " ".join(partials)
    out2 = _hf_pipeline(
        combined,
        max_new_tokens=use_mx,
        min_new_tokens=use_mn,
        num_beams=SUMMARY_NUM_BEAMS,
        length_penalty=SUMMARY_LENGTH_PENALTY,
        no_repeat_ngram_size=SUMMARY_NGRAM_BLOCK,
    )
    return (out2[0].get("summary_text") or "").strip()

def summarize_text(text: str, mn: Optional[int] = None, mx: Optional[int] = None) -> str:
    """
    Prefer HF abstractive (with chunking + per-request token controls).
    Fallback to simple extractive when HF not available.
    """
    text = text.strip()
    if not text:
        return ""
    if len(text) > 120_000:  # guard extreme inputs
        text = text[:120_000]

    if try_init_hf_summarizer():
        try:
            return _summarize_chunked(text, mn=mn, mx=mx)
        except Exception as e:
            log.warning(f"HF summarize failed ({e}); using extractive.")
    # Extractive fallback ignores mn/mx
    return simple_extractive_summarize(text, max_sentences=3)

# =============================
# Formatting (plain/bullets/markdown)
# =============================
def _split_sentences(s: str) -> List[str]:
    return [x.strip() for x in _SENT_SPLIT_RE.split(s.strip()) if x.strip()]

def _to_bullets(summary: str, max_points: int, style: str) -> str:
    sents = _split_sentences(summary)
    sents = sents[: max(1, max_points)]
    bullet = "• " if style == "bullets" else "- "
    return "\n".join(f"{bullet}{s}" for s in sents)

def format_summary(summary: str, fmt: str = "plain", max_points: int = 5) -> str:
    """
    fmt options:
      - "plain": return as-is
      - "bullets": one sentence per line prefixed with '• '
      - "markdown": one sentence per line prefixed with '- '
    """
    fmt = (fmt or "plain").lower()
    if fmt == "plain":
        return summary
    if fmt in {"bullets", "markdown"}:
        return _to_bullets(summary, max_points=max_points, style=fmt)
    return summary  # unknown fmt → plain

# =============================
# FastAPI app + models
# =============================
app = FastAPI(title="Image→Text→Summary (CPU)", version="1.5.0")

# CORS (use env CORS_ALLOW_ORIGINS="*" or comma-separated list)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextIn(BaseModel):
    text: str = Field(min_length=1, description="Raw text to summarize.")

class TextOut(BaseModel):
    summary: str

class OCRResponse(BaseModel):
    text: str
    ocr_chars: int
    ocr_conf: float

class ImageResponse(BaseModel):
    text: str
    ocr_chars: int
    ocr_conf: float
    summary: str

@app.get("/", include_in_schema=False)
def root():
    # Friendly redirect to Swagger UI (you can replace with Vercel UI)
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "storage_dir": str(STORAGE_DIR),
        "ocr": {"lang": OCR_LANG, "version": OCR_VERSION, "min_conf": OCR_MIN_CONF},
        "summary_backend": summarizer_backend_name(),
        "summary_model": SUMMARY_MODEL_ID,
        "formats": ["plain", "bullets", "markdown"],
        "max_points_default": 5,
        "length_controls": {"mn_tokens": "min new tokens", "mx_tokens": "max new tokens"},
    }

@app.post("/infer/text", response_model=TextOut)
def infer_text(
    payload: TextIn,
    fmt: str = Query("plain", pattern="^(plain|bullets|markdown)$"),
    points: int = Query(5, ge=1, le=30, description="Number of bullets if fmt != plain"),
    mn: Optional[int] = Query(None, ge=0, le=512, description="Min new tokens (HF only)"),
    mx: Optional[int] = Query(None, ge=16, le=1024, description="Max new tokens (HF only)"),
):
    """
    Summarize raw text.
    - fmt: output format (plain|bullets|markdown)
    - points: bullet count (if not plain)
    - mn/mx: per-request token hints for HF abstractive
    """
    try:
        summary = summarize_text(payload.text, mn=mn, mx=mx)
        return {"summary": format_summary(summary, fmt, max_points=points)}
    except Exception as e:
        log.exception("Text summarization failed")
        raise HTTPException(status_code=500, detail=f"Summarization error: {e}")

@app.post("/infer/ocr", response_model=OCRResponse)
def infer_ocr(file: UploadFile = File(...)):
    """
    OCR only. Returns raw recognized text + confidence stats.
    """
    try:
        img = _read_image_from_upload(file)
        text, conf, n_chars = run_ocr(img)
        return {"text": text, "ocr_chars": n_chars, "ocr_conf": round(conf, 4)}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("OCR failed")
        raise HTTPException(status_code=500, detail=f"OCR error: {e}")

@app.post("/infer/image", response_model=ImageResponse)
def infer_image(
    file: UploadFile = File(...),
    fmt: str = Query("plain", pattern="^(plain|bullets|markdown)$"),
    points: int = Query(5, ge=1, le=30, description="Number of bullets if fmt != plain"),
    mn: Optional[int] = Query(None, ge=0, le=512, description="Min new tokens (HF only)"),
    mx: Optional[int] = Query(None, ge=16, le=1024, description="Max new tokens (HF only)"),
):
    """
    OCR + summarize.
    - fmt/points control formatting
    - mn/mx steer HF abstractive length when available
    """
    try:
        img = _read_image_from_upload(file)
        text, conf, n_chars = run_ocr(img)
        summary = summarize_text(text, mn=mn, mx=mx) if text else ""
        return {
            "text": text,
            "ocr_chars": n_chars,
            "ocr_conf": round(conf, 4),
            "summary": format_summary(summary, fmt, max_points=points),
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("OCR+Summarization failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
