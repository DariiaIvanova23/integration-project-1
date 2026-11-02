import os
import json
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.model import SimpleAudioCNN
from src.data import SpeechDataset

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=["yes", "no", "up", "down"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="./artifacts")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    dataset = SpeechDataset("./data", args.classes)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    model = SimpleAudioCNN(num_classes=len(args.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    print(f"Початок навчання ({args.epochs} епох)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = round((correct / total) * 100, 2)
    print(f"Final Accuracy: {accuracy}%")

    # Збереження артефактів
    model_path = os.path.join(args.out, "model.pt")
    torch.save(model.state_dict(), model_path)

    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump(args.classes, f)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"accuracy": accuracy, "epochs": args.epochs, "model_version": "1.0.0"}, f)

    print("Артефакти збережено в:", args.out)

if __name__ == "__main__":
    main()