# Image → Text → Summary (CPU‑only, High‑Accuracy)

Production‑grade starter that runs **entirely on CPU** with strong accuracy:

- **OCR**: PaddleOCR (defaults to PP‑OCRv5, server accuracy; angle classifier ON)
- **Captioning**: BLIP base (only used if OCR text is too short/low‑confidence)
- **Summarization**: BART‑Large‑CNN via **ONNX Runtime** (quantized), with tuned beam search

## Quick start (Docker)

```bash
# 1) Clone/unzip this folder
cp .env.sample .env

# 2) Build & run everything
docker compose up --build -d

# 3) Test it with an image
# Replace path/to/local.jpg with a real image on your machine.
curl -F "file=@path/to/local.jpg" http://localhost:8080/jobs | jq

# 4) Poll job status (replace JOB_ID from step 3)
curl http://localhost:8080/jobs/JOB_ID | jq
```

### What happens
1. API saves your image and enqueues a pipeline job.
2. **OCR worker** runs PaddleOCR (angle cls on, PP‑OCRv5) and returns text + mean conf.
3. If text is short or confidence low, **Caption worker** (BLIP base) describes the image.
4. **Summarizer worker** uses **BART‑Large‑CNN ONNX** (int8) with beams for accurate summaries.
5. Results are stored and can be fetched via `GET /jobs/{id}`.

## Accuracy defaults you can tweak (`.env`)

- `OCR_VERSION=PP-OCRv5` (PaddleOCR default; very accurate; CPU‑friendly)
- `OCR_MIN_CHARS=60` and `OCR_MIN_CONF=0.60` → fallback to captioning if below either
- `SUMMARY_MAX_NEW_TOKENS=140`, `SUMMARY_NUM_BEAMS=5`, `SUMMARY_NGRAM_BLOCK=3`, `SUMMARY_LENGTH_PENALTY=1.1`

## Local development (optional)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/common.txt -r requirements/api.txt
uvicorn app.main:app --reload --port 8080
# In other shells, run celery workers (see docker-compose for exact commands)
```

## Notes
- Models are cached in a shared volume so containers don’t re‑download.
- Summarizer is exported & quantized to ONNX at build time for CPU speed with minimal quality loss.
- If you prefer faster summaries, switch to DistilBART in `workers/sum_worker.py` (already supported).

## License
MIT (for this scaffold). Upstream models follow their own licenses.
