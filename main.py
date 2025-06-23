# main.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.quantization import quantize_dynamic
import os

def load_data(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_set = datasets.Caltech101(root="./data", download=True, transform=transform)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True)

def evaluate(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = load_data()

    model_fp32 = models.resnet18(pretrained=True)
    model_fp32.fc = nn.Linear(model_fp32.fc.in_features, 102)
    model_fp32.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_fp32.parameters(), lr=0.001)

    model_fp32.train()
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model_fp32(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        break  # Train only one batch for demo purposes

    acc_fp32 = evaluate(model_fp32, dataloader, device)
    print(f"ResNet18 Accuracy (FP32): {acc_fp32:.4f}")

    model_fp32.cpu()
    model_int8 = quantize_dynamic(model_fp32, {nn.Linear}, dtype=torch.qint8)

    acc_int8 = evaluate(model_int8, dataloader, torch.device("cpu"))
    print(f"ResNet18 Accuracy (Quantized INT8): {acc_int8:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model_int8.state_dict(), "models/resnet18_quantized.pth")

if __name__ == "__main__":
    main()
