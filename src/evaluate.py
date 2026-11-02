import os
import json
import torch
from torch.utils.data import DataLoader
from src.model import SimpleAudioCNN
from src.data import SpeechDataset

def main():
    art_dir = os.environ.get("ARTIFACTS_DIR", "./artifacts")
    with open(os.path.join(art_dir, "labels.json")) as f:
        classes = json.load(f)

    dataset = SpeechDataset("./data", classes)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = SimpleAudioCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(os.path.join(art_dir, "model.pt"), map_location="cpu"))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = round((correct / total) * 100, 2)
    print(f"Evaluated Accuracy: {acc}%")

if __name__ == "__main__":
    main()