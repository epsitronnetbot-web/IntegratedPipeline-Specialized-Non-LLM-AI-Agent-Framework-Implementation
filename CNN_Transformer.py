import math
import os
import kagglehub
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path
import sys

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    """Conv -> BN -> Activation, optionally downsampling with stride=2."""

    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class ResidualBlock(nn.Module):
    """Two conv layers with a skip connection. Downsamples if stride != 1."""

    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.block1 = ConvBlock(in_ch, out_ch, stride=stride, dropout=dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.block1(x)
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class RobustCNN(nn.Module):
    """
    A ResNet-style CNN sized for small/medium images (e.g. 32x32 or 64x64).

    Stages progressively downsample spatial resolution while increasing
    channel depth, followed by global average pooling and a linear head.
    """

    def __init__(self, in_channels=3, num_classes=10, base_width=64,
                 blocks_per_stage=(2, 2, 2), dropout=0.1, head_dropout=0.3):
        super().__init__()

        self.stem = ConvBlock(in_channels, base_width, stride=1)

        widths = [base_width * (2 ** i) for i in range(len(blocks_per_stage))]
        stages = []
        in_ch = base_width
        for stage_idx, (width, n_blocks) in enumerate(zip(widths, blocks_per_stage)):
            for block_idx in range(n_blocks):
                # Downsample at the start of every stage except the first
                stride = 2 if (block_idx == 0 and stage_idx != 0) else 1
                stages.append(ResidualBlock(in_ch, width, stride=stride, dropout=dropout))
                in_ch = width
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(head_dropout),
            nn.Linear(in_ch, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        return self.head(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

model = RobustCNN(
    in_channels=3,
    num_classes=2,          # organic vs recyclable, bukan 10 seperti CIFAR
    base_width=64,
    blocks_per_stage=(2, 2, 2),
    dropout=0.1,
    head_dropout=0.3
).to(device)
IMG_SIZE = 64

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),      # augmentasi ringan, gambar sampah tidak sensitif orientasi
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats, umum dipakai
                          std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

base_path = kagglehub.dataset_download("techsash/waste-classification-data")
train_dir = os.path.join(base_path, "DATASET", "TRAIN")
test_dir = os.path.join(base_path, "DATASET", "TEST")

full_train = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transform)

# Split train jadi train + validation (dataset TRAIN/TEST folder biasanya tidak ada val)
val_size = int(0.15 * len(full_train))
train_size = len(full_train) - val_size
train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()   # cocok untuk num_classes=2 dengan label integer (bukan one-hot)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="training", leave=False)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=running_loss / total, acc=correct / total)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

def main():
    num_epochs = 30
    best_val_loss = float("inf")
    patience, patience_counter = 5, 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Simpan model terbaik
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "cnn_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load("cnn_model.pth"))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()
