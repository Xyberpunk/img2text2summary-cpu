import os
from app.celery_app import celery_app

# Try ONNXRuntime via Optimum first
USE_ONNX = os.getenv("SUMMARY_USE_ONNX", "1") == "1"

summary_cfg = {
    "max_new_tokens": int(os.getenv("SUMMARY_MAX_NEW_TOKENS","140")),
    "min_new_tokens": int(os.getenv("SUMMARY_MIN_NEW_TOKENS","40")),
    "num_beams": int(os.getenv("SUMMARY_NUM_BEAMS","5")),
    "no_repeat_ngram_size": int(os.getenv("SUMMARY_NGRAM_BLOCK","3")),
    "length_penalty": float(os.getenv("SUMMARY_LENGTH_PENALTY","1.1")),
}

onnx_dir = "/models/bart-onnx"

ort_model = None
tokenizer = None

def _load_onnx():
    global ort_model, tokenizer
    if ort_model is not None:
        return True
    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(onnx_dir)
        tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
        return True
    except Exception as e:
        return False

pipe = None
def _load_fallback():
    global pipe
    if pipe is not None: return True
    from transformers import pipeline
    # Balanced fallback: DistilBART (fast/accurate on CPU)
    pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    return True

def _summarize(text: str) -> str:
    # truncate very long inputs to keep CPU time reasonable
    text = text.strip()
    if USE_ONNX and _load_onnx():
        inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=1024)
        import torch
        with torch.no_grad():
            out = ort_model.generate(
                **inputs,
                max_new_tokens=summary_cfg["max_new_tokens"],
                min_new_tokens=summary_cfg["min_new_tokens"],
                num_beams=summary_cfg["num_beams"],
                no_repeat_ngram_size=summary_cfg["no_repeat_ngram_size"],
                length_penalty=summary_cfg["length_penalty"],
                do_sample=False,
                early_stopping=True
            )
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()
    else:
        _load_fallback()
        out = pipe(
            text[:4000],
            max_length=summary_cfg["max_new_tokens"],
            min_length=max(20, summary_cfg["min_new_tokens"]),
            do_sample=False,
            num_beams=summary_cfg["num_beams"],
            no_repeat_ngram_size=summary_cfg["no_repeat_ngram_size"],
            length_penalty=summary_cfg["length_penalty"]
        )
        return out[0]["summary_text"].strip()

@celery_app.task(name="sum.run")
def sum_run(text_for_summary: str, source: str):
    summary = _summarize(text_for_summary)
    return {"summary": summary, "source": source}
