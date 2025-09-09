import os
from PIL import Image
from app.celery_app import celery_app
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

DEVICE = "cpu"
MODEL_ID = "Salesforce/blip-image-captioning-base"

processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
model.to(DEVICE)
model.eval()

@celery_app.task(name="caption.run")
def caption_run(image_path: str):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=4,
            do_sample=False
        )
    caption = processor.decode(out[0], skip_special_tokens=True)
    return {"caption": caption.strip()}
