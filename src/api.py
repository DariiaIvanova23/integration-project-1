import os
import io
import time
import json
import soundfile as sf
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from src.model import SimpleAudioCNN
from src.data import transform, pad_or_trim

app = FastAPI(title="KWS Production API", version="1.0.0")

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.pt")
LABELS_PATH = os.path.join(os.path.dirname(MODEL_PATH), "labels.json")
METRICS_PATH = os.path.join(os.path.dirname(MODEL_PATH), "metrics.json")

# Завантажуємо артефакти один раз при старті
with open(LABELS_PATH) as f:
    LABELS = json.load(f)

with open(METRICS_PATH) as f:
    METRICS = json.load(f)

device = torch.device("cpu")
model = SimpleAudioCNN(num_classes=len(LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    size_kb = round(os.path.getsize(MODEL_PATH) / 1024, 2)
    return {
        "architecture": "SimpleAudioCNN (2D Conv)",
        "classes": LABELS,
        "model_size_kb": size_kb,
        "accuracy": METRICS.get("accuracy"),
        "model_version": METRICS.get("model_version", "1.0.0")
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Формат файлу має бути суто .wav")

    content = await file.read()
    try:
        data, sr = sf.read(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Некоректний аудіофайл: {str(e)}")

    start_time = time.perf_counter()

    waveform = torch.tensor(data, dtype=torch.float32)
    if waveform.ndim > 1:
        waveform = waveform[:, 0]
    waveform = waveform.unsqueeze(0)
    waveform = pad_or_trim(waveform)
    spec = transform(waveform).unsqueeze(0)

    with torch.no_grad():
        logits = model(spec)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    latency_ms = (time.perf_counter() - start_time) * 1000

    prob_dict = {LABELS[i]: round(probs[i].item(), 3) for i in range(len(LABELS))}

    return {
        "label": LABELS[pred_idx],
        "confidence": round(probs[pred_idx].item(), 3),
        "probabilities": prob_dict,
        "latency_ms": round(latency_ms, 2),
        "model_version": METRICS.get("model_version", "1.0.0")
    }