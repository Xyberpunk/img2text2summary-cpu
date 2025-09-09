# app/infer.py
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")  # avoid torchvision import

import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2

log = logging.getLogger("img2text2summary.infer")

# -----------------------------
# Config (env + sane defaults)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# OCR config
OCR_LANG = os.getenv("OCR_LANG", "en")
OCR_VERSION_RAW = os.getenv("OCR_VERSION", "PP-OCRv4")
OCR_MIN_CONF = float(os.getenv("OCR_MIN_CONF", "0.60"))

# Summarization config (defaults; can be overridden per-request via mn/mx/points)
SUMMARY_MODEL_ID = os.getenv("SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")
SUMMARY_MAX_NEW_TOKENS = int(os.getenv("SUMMARY_MAX_NEW_TOKENS", "140"))
SUMMARY_MIN_NEW_TOKENS = int(os.getenv("SUMMARY_MIN_NEW_TOKENS", "40"))
SUMMARY_NUM_BEAMS = int(os.getenv("SUMMARY_NUM_BEAMS", "5"))
SUMMARY_LENGTH_PENALTY = float(os.getenv("SUMMARY_LENGTH_PENALTY", "1.1"))
SUMMARY_NGRAM_BLOCK = int(os.getenv("SUMMARY_NGRAM_BLOCK", "3"))

# Preprocess toggles
PRE_DENOISE = os.getenv("PRE_DENOISE", "1") == "1"
PRE_DESKEW = os.getenv("PRE_DESKEW", "1") == "1"
PRE_CONTRAST = os.getenv("PRE_CONTRAST", "1") == "1"

# Globals (lazy-init)
_ocr_backend = None
_hf_pipeline = None
_summarizer_backend: Optional[str] = None  # "hf" | "simple" | None

# -----------------------------
# Utils
# -----------------------------
def _normalize_ocr_version(v: str) -> str:
    """PaddleOCR 2.7.x supports up to PP-OCRv4. Map variants to canonical form."""
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
        log.warning("OCR_VERSION=PP-OCRv5 is not supported by paddleocr==2.7.x; using PP-OCRv4.")
        return "PP-OCRv4"
    return "PP-OCRv4"

OCR_VERSION = _normalize_ocr_version(OCR_VERSION_RAW)

# -----------------------------
# OCR
# -----------------------------
def get_ocr_backend():
    global _ocr_backend
    if _ocr_backend is None:
        from paddleocr import PaddleOCR  # heavy import
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


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Light, CPU-friendly preprocessing to help OCR."""
    out = img.copy()

    if PRE_DESKEW:
        try:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thr == 0))
            if coords.size > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
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


def read_image_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    return img


def run_ocr(img: np.ndarray) -> Tuple[str, float, int]:
    """Return (joined_text, avg_conf, char_count) with reading-order sorting."""
    ocr = get_ocr_backend()
    pre = preprocess_image(img)
    result = ocr.ocr(pre, cls=True)

    lines: List[str] = []
    confs: List[float] = []
    for page in result:
        # Sort top→bottom, then left→right
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

# -----------------------------
# Summarization
# -----------------------------
_STOPWORDS = set("""
a an and are as at be by for from has have he her hers him his i in is it its of on or our she that the their them they this to was we were will with you your
""".strip().split())
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def simple_extractive_summarize(text: str, max_sentences: int = 5) -> str:
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if not sents:
        return text.strip()
    freqs = {}
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    for w in words:
        if w in _STOPWORDS:
            continue
        freqs[w] = freqs.get(w, 0) + 1
    if not freqs:
        return " ".join(sents[:max_sentences])
    scores = []
    for idx, s in enumerate(sents):
        sw = re.findall(r"[A-Za-z0-9']+", s.lower())
        score = sum(freqs.get(w, 0) for w in sw) / (len(sw) + 1e-6)
        # small bonus for earlier sentences to improve coherence
        score += max(0.0, 1.0 - (idx / max(1, len(sents))))
        scores.append((score, idx, s))
    top = sorted(scores, key=lambda x: (-x[0], x[1]))[:max_sentences]
    top_sorted = [s for (_, _, s) in sorted(top, key=lambda x: x[1])]
    return " ".join(top_sorted)

def try_init_hf_summarizer() -> bool:
    """Attempt to build a HF pipeline on CPU. Falls back to extractive."""
    global _hf_pipeline, _summarizer_backend
    if _hf_pipeline is not None:
        return True
    if _summarizer_backend == "simple":
        return False
    try:
        from transformers import pipeline  # type: ignore
        log.info(f"Initializing HF summarizer: {SUMMARY_MODEL_ID} (CPU)…")
        _hf_pipeline = pipeline(
            "summarization",
            model=SUMMARY_MODEL_ID,
            tokenizer=SUMMARY_MODEL_ID,
            device=-1,      # CPU
            framework="pt", # requires torch; if missing, will raise
        )
        _summarizer_backend = "hf"
        log.info("HF summarizer ready.")
        return True
    except Exception as e:
        log.warning(f"HF summarizer not available ({e}); using simple extractive).")
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

def _hf_summarize_once(text: str, mn: int, mx: int) -> str:
    """Single-pass HF summary honoring token hints."""
    assert _hf_pipeline is not None
    mx = max(16, int(mx))
    mn = max(0, int(min(mn, mx // 2)))
    out = _hf_pipeline(
        text,
        max_new_tokens=mx,
        min_new_tokens=mn,
        num_beams=SUMMARY_NUM_BEAMS,
        length_penalty=SUMMARY_LENGTH_PENALTY,
        no_repeat_ngram_size=SUMMARY_NGRAM_BLOCK,
    )
    return (out[0].get("summary_text") or "").strip()

def _summarize_chunked(text: str, mn: int, mx: int) -> str:
    """Map-reduce style summary over token chunks, then a final pass (respects mn/mx)."""
    assert _hf_pipeline is not None
    # Per-chunk smaller targets; final pass uses requested mn/mx
    chunk_mx = max(32, min(mx, 160))
    chunk_mn = max(0, min(mn, chunk_mx // 2))

    partials: List[str] = []
    for piece in _chunks_by_tokens(text, max_tokens=800, overlap=120):
        out = _hf_pipeline(
            piece,
            max_new_tokens=chunk_mx,
            min_new_tokens=chunk_mn,
            num_beams=SUMMARY_NUM_BEAMS,
            length_penalty=SUMMARY_LENGTH_PENALTY,
            no_repeat_ngram_size=SUMMARY_NGRAM_BLOCK,
        )
        partials.append((out[0].get("summary_text") or "").strip())

    combined = " ".join(partials)
    return _hf_summarize_once(combined, mn, mx)

def summarize_text(
    text: str,
    mn: Optional[int] = None,
    mx: Optional[int] = None,
    points: Optional[int] = None,
) -> str:
    """
    Produce a summary; if HF available, use mn/mx to steer length.
    If only extractive is available, use 'points' as sentence budget.
    """
    text = (text or "").strip()
    if not text:
        return ""

    # Light guard for extremely long inputs
    if len(text) > 120_000:
        text = text[:120_000]

    # Defaults if caller didn't override
    _mx = int(mx) if mx is not None else SUMMARY_MAX_NEW_TOKENS
    _mn = int(mn) if mn is not None else min(SUMMARY_MIN_NEW_TOKENS, _mx // 2)

    if try_init_hf_summarizer():
        try:
            # Use chunked path for long inputs, single pass for short ones
            if len(text) > 4000:
                return _summarize_chunked(text, _mn, _mx)
            return _hf_summarize_once(text, _mn, _mx)
        except Exception as e:
            log.warning(f"HF summarize failed ({e}); fallback to extractive.")

    # Extractive fallback
    target = points if (points and points > 0) else 5
    return simple_extractive_summarize(text, max_sentences=target)

# -----------------------------
# Bullets formatting
# -----------------------------
def _sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT_RE.split(text.strip())
    sents = [s.strip().rstrip(" .") for s in parts if s.strip()]
    out: List[str] = []
    for s in sents:
        for chunk in re.split(r"\s*;\s*|\s*[–—]\s*", s):
            if len(chunk.split()) >= 3:
                out.append(chunk.strip())
    return out

def make_bullets(text: str, max_items: int = 6) -> List[str]:
    sents = _sentences(text)
    if not sents and text.strip():
        return [text.strip()]
    return sents[:max(1, max_items)]

def bullets_markdown(bullets: List[str]) -> str:
    return "\n".join(f"- {b}" for b in bullets)

def format_summary(summary: str, fmt: str = "plain", points: Optional[int] = None) -> str:
    """
    fmt: plain | bullets | markdown
    points: desired number of bullet lines (defaults to 5)
    """
    fmt = (fmt or "plain").lower()
    if fmt == "plain":
        return summary
    max_items = points if (points and points > 0) else 5
    bullets = make_bullets(summary, max_items=max_items)
    if fmt == "markdown":
        return bullets_markdown(bullets)
    # default 'bullets'
    return "\n".join(f"• {b}" for b in bullets)

# Expose a small state view for /health
def summarizer_backend_name() -> str:
    return _summarizer_backend or "init_on_first_use"
