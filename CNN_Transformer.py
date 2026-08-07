"""
Robust CNN for image classification in PyTorch.

Design choices baked in for robustness:
  - Conv blocks: Conv -> BatchNorm -> Activation -> (Dropout)
  - Residual connections to ease optimization in deeper stacks
  - Global Average Pooling instead of a huge flatten+FC (fewer params, less overfitting)
  - Kaiming weight initialization
  - Label smoothing in the loss (reduces overconfidence)
  - Data augmentation (flip, crop, color jitter, optional mixup)
  - LR scheduling (cosine annealing with warmup) + gradient clipping
  - Mixed precision training (faster, lower memory, same accuracy)
  - Early stopping + checkpointing on best validation metric

Adjust `NUM_CLASSES`, `IN_CHANNELS`, and image size assumptions (32x32 by
default, like CIFAR-10/100) to fit your dataset.
"""

import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import json

from AbstractIntegratedModule import IntegratedPipeline
from AbstractIntegratedModule import Transformer


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Data (CIFAR-10 example — swap for your own dataset/ImageFolder as needed)
# --------------------------------------------------------------------------- #
# On Windows, DataLoader worker processes have real startup overhead
# (each is a fresh Python process, often slowed further by antivirus
# scanning). num_workers=2 is a safer default there; num_workers=0
# avoids multiprocessing entirely if things still seem slow to start.
def get_dataloaders(data_dir="./data", batch_size=128, num_workers=2):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25),  # cutout-style regularization
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR100(data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def get_custom_dataloaders(train_dir, val_dir, image_size=64, batch_size=128,
                            num_workers=4):
    """
    Use your own images instead of CIFAR-10.

    Expected folder layout (ImageFolder standard):
        train_dir/
            class_a/  img1.jpg  img2.jpg  ...
            class_b/  img1.jpg  ...
        val_dir/
            class_a/  ...
            class_b/  ...

    Class names/order are inferred automatically from the subfolder names.
    Returns: train_loader, val_loader, class_names (list, index = label id)
    """
    # ImageNet mean/std work reasonably well as a default for natural photos.
    # If your images are very different (e.g. grayscale, medical, satellite),
    # compute your own mean/std over the training set instead.
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    val_set = datasets.ImageFolder(val_dir, transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_set.classes


# --------------------------------------------------------------------------- #
# Mixup (optional but effective regularizer — toggle via USE_MIXUP)
# --------------------------------------------------------------------------- #
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# --------------------------------------------------------------------------- #
# Training / evaluation loop
# --------------------------------------------------------------------------- #
class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf
        self.counter = 0

    def step(self, metric):
        """Returns True if training should stop."""
        if metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, optimizer, criterion, device, scaler,
                     use_mixup=True, grad_clip=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="training", leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            if use_mixup:
                x_mixed, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
                out = model(x_mixed)
                loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            else:
                out = model(x)
                loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        pbar.set_postfix(loss=total_loss / total, acc=correct / total)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def transformer_training(pipeline, img, class_names):
    model_transformer = Transformer(
                    vocab_size=1,
                    d_model=pipeline.transformer_d_model,
                    n_heads=pipeline.transformer_heads,
                    num_classes=len(class_names)
                )
    X = transform(img).unsqueeze(0).to(device)  # add batch dimension
    sequence_inputs = pipeline._features_to_sequence(X, d_model=pipeline.transformer_d_model)
    AME = pipeline.model2.AME_Encoder(X)

    model_transformer.train(sequence_inputs, AME=AME, embedded=True)
    tf = model_transformer

    json_data = {
        'token_embedding': tf.token_embedding,
        'pos_embedding': tf.pos_embedding,
        'W_q': tf.W_q,
        'W_k': tf.W_k,
        'W_v': tf.W_v,
        'W_q_fixed': tf.W_q_fixed,
        'W_k_fixed': tf.W_k_fixed,
        'W_v_fixed': tf.W_v_fixed,
        'W_o': tf.W_o,
        'ffn1': tf.ffn1,
        'ffn2': tf.ffn2,
        'ln1_scale': tf.ln1_scale,
        'ln1_shift': tf.ln1_shift,
        'ln2_scale': tf.ln2_scale,
        'ln2_shift': tf.ln2_shift,
        'output': tf.output,
        'output_bias': tf.output_bias
    }
    with open("transformer_weights.json", "w") as file:
        json.dump(data, file, indent=4)  # indent adds readable formatting


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Toggle this: True = use your own image folders, False = CIFAR-10 ----
    USE_CUSTOM_DATA = False

    EPOCHS = 100
    BATCH_SIZE = 128
    LR = 0.1
    WEIGHT_DECAY = 5e-4
    WARMUP_EPOCHS = 5
    USE_MIXUP = True
    CKPT_PATH = "best_model.pt"

    if USE_CUSTOM_DATA:
        IMAGE_SIZE = 32  # match to your images; bigger = more compute
        train_loader, test_loader, class_names = get_custom_dataloaders(
            train_dir="my_data/train",
            val_dir="my_data/val",
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
        )
        NUM_CLASSES = len(class_names)
        print(f"Found {NUM_CLASSES} classes: {class_names}")
    else:
        NUM_CLASSES = 100
        train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # ---- Model size — edit these to control param count / speed ----
    MODEL_CONFIG = {
        "in_channels": 3,
        "base_width": 16,          # try 16 (~100K params), 28 (~530K), 64 (~2.8M default)
        "blocks_per_stage": (1, 2, 1),
    }

    model = RobustCNN(num_classes=NUM_CLASSES, **MODEL_CONFIG).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                                 weight_decay=WEIGHT_DECAY, nesterov=True)

    # Cosine schedule with linear warmup
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    early_stopper = EarlyStopper(patience=15)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_mixup=USE_MIXUP
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"lr {optimizer.param_groups[0]['lr']:.5f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": NUM_CLASSES,
                "class_names": class_names if USE_CUSTOM_DATA else
                    ["airplane", "automobile", "bird", "cat", "deer",
                     "dog", "frog", "horse", "ship", "truck"],
                "model_config": MODEL_CONFIG,
                "image_size": IMAGE_SIZE if USE_CUSTOM_DATA else 32,
            }, CKPT_PATH)

        if early_stopper.step(val_acc):
            print(f"Early stopping at epoch {epoch+1}. Best val_acc: {best_acc:.4f}")
            break

    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")
    print(f"Best checkpoint saved to: {CKPT_PATH}")

    print('=== Transformer Training ===')
    transformer_training(pipeline, img, class_names)  # img should be defined or passed appropriately


if __name__ == "__main__":
    main()
