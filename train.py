
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CatDogCNN

# Mac GPU available hai toh use karo
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Training on:", device)

# Images ko prepare karo
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(
    "data/train", transform=train_transform
)

val_data = datasets.ImageFolder(
    "data/val", transform=val_transform
)

train_loader = DataLoader(
    train_data, batch_size=32, shuffle=True
)

val_loader = DataLoader(
    val_data, batch_size=32, shuffle=False
)

print("Classes:", train_data.class_to_idx)

# CNN model create karo
model = CatDogCNN().to(device)

# Loss aur optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

Path("saved_models").mkdir(exist_ok=True)

best_accuracy = -1.0

# 10 epochs training
for epoch in range(10):

    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(train_data)

    print(
        f"Epoch {epoch+1}/10 | "
        f"Loss: {avg_loss:.4f} | "
        f"Val Accuracy: {accuracy:.2%}"
    )

    # Best model save karo
    if accuracy > best_accuracy:
        best_accuracy = accuracy

        torch.save({
            "model_state": model.state_dict(),
            "class_to_idx": train_data.class_to_idx,
            "image_size": 128
        }, "saved_models/cat_dog_model.pth")

print("Best Validation Accuracy:", best_accuracy)
print("Model saved successfully!")
