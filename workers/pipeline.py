import os, time, traceback
from datetime import datetime
from app.celery_app import celery_app
from app.db import get_session, Job
from statistics import mean

OCR_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", "60"))
OCR_MIN_CONF = float(os.getenv("OCR_MIN_CONF", "0.60"))

@celery_app.task(name="pipeline.process_job")
def process_job(job_id: str):
    # Update status
    from celery.result import AsyncResult
    with get_session() as s:
        job = s.get(Job, job_id); 
        if not job: return
        job.status = "processing"; job.updated_at = datetime.utcnow(); s.add(job); s.commit()

    try:
        # 1) OCR
        ocr_res = celery_app.send_task("ocr.run", args=[job.image_path]).get(timeout=120)
        text = ocr_res.get("text","").strip()
        conf = float(ocr_res.get("mean_conf", 0.0))
        chars = len(text)

        # Decide path
        if chars < OCR_MIN_CHARS or conf < OCR_MIN_CONF:
            # 2) Caption fallback
            cap = celery_app.send_task("caption.run", args=[job.image_path]).get(timeout=120)
            text_for_summary = f"Image description: {cap.get('caption','').strip()}\nSummarize the key information in 2-3 sentences."
            source = "caption"
        else:
            text_for_summary = text
            source = "ocr"

        # 3) Summarize
        summ = celery_app.send_task("sum.run", args=[text_for_summary, source]).get(timeout=180)

        with get_session() as s:
            job = s.get(Job, job_id)
            job.status = "done"
            job.source = source
            job.ocr_chars = chars
            job.ocr_conf = conf
            job.summary = summ.get("summary","")
            job.updated_at = datetime.utcnow()
            s.add(job); s.commit()

    except Exception as e:
        with get_session() as s:
            job = s.get(Job, job_id)
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.updated_at = datetime.utcnow()
            s.add(job); s.commit()
        traceback.print_exc()
