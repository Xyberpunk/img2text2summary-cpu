cd /mnt/d/img2text2summary-cpu
cat > app/celery_app.py <<'PY'
import os
import logging
from celery import Celery
from kombu import Queue  # NOTE: don't pass Exchange objects into defaults

log = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

celery_app = Celery(
    "img2text2summary",
    broker=REDIS_URL,
    backend=RESULT_BACKEND,
    include=[
        "workers.pipeline",
        "workers.ocr_worker",
        "workers.caption_worker",
        "workers.sum_worker",
    ],
)

celery_app.conf.update(
    # serializers
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # timing / reliability
    timezone=os.getenv("TZ", "Asia/Kolkata"),
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=int(os.getenv("CELERY_PREFETCH", "1")),
    broker_transport_options={
        "visibility_timeout": int(os.getenv("CELERY_VISIBILITY_TIMEOUT", "1800")),
    },

    # ----- queues & routing (use STRINGS for defaults) -----
    task_default_exchange="orchestrator",
    task_default_exchange_type="direct",
    task_default_queue="orchestrator",
    task_default_routing_key="orchestrator",

    task_queues=[
        Queue("orchestrator", routing_key="orchestrator"),
        Queue("ocr",          routing_key="ocr"),
        Queue("caption",      routing_key="caption"),
        Queue("summary",      routing_key="summary"),
    ],

    task_routes={
        "pipeline.process_job": {"queue": "orchestrator", "routing_key": "orchestrator"},
        "ocr.run":              {"queue": "ocr",          "routing_key": "ocr"},
        "caption.run":          {"queue": "caption",      "routing_key": "caption"},
        "sum.run":              {"queue": "summary",      "routing_key": "summary"},
    },

    result_expires=int(os.getenv("CELERY_RESULT_EXPIRES", "3600")),
)

def enqueue_job(job_id: str):
    """Helper for API to trigger the pipeline."""
    r = celery_app.send_task(
        "pipeline.process_job", args=[job_id],
        queue="orchestrator", routing_key="orchestrator"
    )
    log.info("Enqueued job %s -> orchestrator task_id=%s", job_id, r.id)
    return r

if __name__ == "__main__":
    celery_app.start()
PY
