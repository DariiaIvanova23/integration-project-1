import io
import time
import soundfile as sf
import torch
import torchaudio.transforms as T
from fastapi import FastAPI, File, UploadFile
from train import SimpleAudioCNN, LABELS, pad_or_trim

app = FastAPI(title="Audio Commands API")

device = torch.device("cpu")
model = SimpleAudioCNN()
# Завантажуємо навчені ваги
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

transform = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=32)

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    # 1. Зчитування вхідного аудіо через soundfile
    content = await file.read()
    data, sample_rate = sf.read(io.BytesIO(content))
    
    waveform = torch.tensor(data, dtype=torch.float32)
    # Якщо аудіо має кілька каналів (стерео) — беремо лише один
    if waveform.ndim > 1:
        waveform = waveform[:, 0]
    waveform = waveform.unsqueeze(0)

    # Приводимо розмір до 1 секунди
    waveform = pad_or_trim(waveform)
    spec = transform(waveform).unsqueeze(0)

    # 2. Вимірювання Latency (час інференсу)
    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model(spec)
        pred_idx = logits.argmax(dim=1).item()
    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        "prediction": LABELS[pred_idx],
        "latency_ms": round(latency_ms, 2)
    }

@app.get("/")
def root():
    return {"message": "Audio Model API працює! Відкрийте /docs для перевірки."}