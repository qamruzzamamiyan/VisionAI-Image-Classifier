
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model_v2 import CatDogCNNV2

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Mac M4 Pro GPU
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Training Device:", device)

# Training settings
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 0.001
IMAGE_SIZE = 128

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),
    transforms.ToTensor()
])

# Validation images
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# Load dataset
train_data = datasets.ImageFolder(
    "data/train",
    transform=train_transform
)

val_data = datasets.ImageFolder(
    "data/val",
    transform=val_transform
)

assert train_data.class_to_idx == val_data.class_to_idx

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Classes:", train_data.class_to_idx)
print("Training images:", len(train_data))
print("Validation images:", len(val_data))

# Create CNN V2 model
model = CatDogCNNV2().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

# Create output folders
Path("saved_models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

history = []
best_val_loss = float("inf")
best_epoch = 0
bad_epochs = 0

# Start training
for epoch in range(1, EPOCHS + 1):

    model.train()

    train_loss_sum = 0
    train_correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * images.size(0)

        train_correct += (
            outputs.argmax(1) == labels
        ).sum().item()

    train_loss = train_loss_sum / len(train_data)
    train_acc = train_correct / len(train_data)

    # Validation
    model.eval()

    val_loss_sum = 0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * images.size(0)

            val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()

    val_loss = val_loss_sum / len(val_data)
    val_acc = val_correct / len(val_data)

    current_lr = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "learning_rate": current_lr
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2%} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2%} | "
        f"LR: {current_lr:.6f}",
        flush=True
    )

    # Save best model
    if val_loss < best_val_loss - 0.0001:
        best_val_loss = val_loss
        best_epoch = epoch
        bad_epochs = 0

        torch.save({
            "model_state": model.state_dict(),
            "class_to_idx": train_data.class_to_idx,
            "image_size": IMAGE_SIZE,
            "epoch": epoch,
            "val_accuracy": val_acc,
            "val_loss": val_loss
        }, "saved_models/cat_dog_v2.pth")

        print("New best model saved!")

    else:
        bad_epochs += 1

    scheduler.step(val_loss)

    # Early stopping
    if bad_epochs >= PATIENCE:
        print("Early stopping triggered.")
        break

# Save training history
with open("results/v2_history.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=history[0].keys()
    )

    writer.writeheader()
    writer.writerows(history)

print("\nTraining completed!")
print("Best epoch:", best_epoch)
print("Best validation loss:", round(best_val_loss, 4))
print("Saved model: saved_models/cat_dog_v2.pth")
