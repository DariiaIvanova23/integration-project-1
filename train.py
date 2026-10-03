import os
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

LABELS = ['yes', 'no', 'up', 'down']
DEVICE = torch.device("cpu")

# Перетворення у Mel-Spectrogram
transform = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=32)

def pad_or_trim(waveform):
    if waveform.shape[1] < 16000:
        return torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
    return waveform[:, :16000]

# Кастомний клас для завантаження тільки потрібних файлів напряму через soundfile
class SimpleSpeechCommands(Dataset):
    def __init__(self, root_dir):
        self.file_list = []
        base_path = os.path.join(root_dir, "SpeechCommands", "speech_commands_v0.02")
        # Якщо вкладеної папки немає, шукаємо прямо в root_dir
        if not os.path.exists(base_path):
            base_path = root_dir

        for label in LABELS:
            folder = os.path.join(base_path, label)
            if os.path.exists(folder):
                for f in os.listdir(folder)[:500]: # беремо по 500 записів кожного слова
                    if f.endswith(".wav"):
                        self.file_list.append((os.path.join(folder, f), label))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        data, sr = sf.read(path)
        waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        waveform = pad_or_trim(waveform)
        spec = transform(waveform)
        return spec, LABELS.index(label)

class SimpleAudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, len(LABELS))
        )
    def forward(self, x):
        return self.net(x)

def main():
    print("Зчитування даних (класи: yes, no, up, down)...")
    dataset = SimpleSpeechCommands("./data")
    if len(dataset) == 0:
        print("Помилка: Аудіо файли не знайдені в папці ./data! Перевірте шлях.")
        return

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    model = SimpleAudioCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    print(f"Початок навчання на {len(dataset)} аудіозаписах (3 епохи)...")
    for epoch in range(1, 4):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        print(f"Епоха {epoch}/3 завершена.")

    # Оцінка точності
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = (correct / total) * 100
    print("\n==========================================")
    print(f"Результат тестування (Accuracy): {accuracy:.2f}%")
    print("==========================================")

    torch.save(model.state_dict(), "model.pth")
    size_kb = os.path.getsize("model.pth") / 1024
    print(f"Модель збережено в 'model.pth'! Розмір: {size_kb:.2f} KB\n")

if __name__ == "__main__":
    main()