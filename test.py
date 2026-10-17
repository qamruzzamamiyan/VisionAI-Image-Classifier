
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from model import CatDogCNN

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Saved model load karo
checkpoint = torch.load(
    "saved_models/cat_dog_model.pth",
    map_location=device,
    weights_only=True
)

model = CatDogCNN().to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Same preprocessing as validation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

test_data = datasets.ImageFolder(
    "data/test",
    transform=transform
)

# Ensure label order matches training
assert test_data.class_to_idx == checkpoint["class_to_idx"]

test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False
)

actual = []
predicted = []

# Testing
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        actual.extend(labels.tolist())
        predicted.extend(predictions.cpu().tolist())

# Results
accuracy = accuracy_score(actual, predicted)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    actual,
    predicted,
    target_names=test_data.classes,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(actual, predicted))
