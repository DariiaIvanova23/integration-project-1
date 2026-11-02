import os
import torch
import soundfile as sf
import torchaudio.transforms as T
from torch.utils.data import Dataset

transform = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=32)

def pad_or_trim(waveform):
    if waveform.shape[1] < 16000:
        return torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
    return waveform[:, :16000]

class SpeechDataset(Dataset):
    def __init__(self, root_dir, classes):
        self.classes = classes
        self.items = []

        # Завантажуємо реальні файли з папки samples
        for idx, label in enumerate(classes):
            path = os.path.join("samples", f"{label}.wav")
            if os.path.exists(path):
                data, _ = sf.read(path)
                wave = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
                if wave.ndim > 2:
                    wave = wave[0]
                wave = pad_or_trim(wave)
                # Робимо копії з різною гучністю та легким шумом (аугментація)
                for _ in range(50):
                    augmented = wave * (0.7 + 0.6 * torch.rand(1)) + 0.005 * torch.randn_like(wave)
                    self.items.append((augmented, idx))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wave, label_idx = self.items[idx]
        spec = transform(wave)
        return spec, label_idx