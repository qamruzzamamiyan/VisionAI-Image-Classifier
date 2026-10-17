
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from model_v2 import CatDogCNNV2

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Load trained V2 model
checkpoint = torch.load(
    "saved_models/cat_dog_v2.pth",
    map_location=device,
    weights_only=True
)

model = CatDogCNNV2().to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Same image preprocessing used in validation
transform = transforms.Compose([
    transforms.Resize((
        checkpoint["image_size"],
        checkpoint["image_size"]
    )),
    transforms.ToTensor()
])

test_data = datasets.ImageFolder(
    "data/test",
    transform=transform
)

assert test_data.class_to_idx == checkpoint["class_to_idx"]

test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False
)

actual = []
predicted = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        actual.extend(labels.tolist())
        predicted.extend(predictions.cpu().tolist())

accuracy = accuracy_score(actual, predicted)

print("\n====== CNN V2 TEST RESULTS ======")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    actual,
    predicted,
    target_names=test_data.classes,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(actual, predicted))

print("\n====== V1 vs V2 ======")
print("V1 Test Accuracy: 71.20%")
print(f"V2 Test Accuracy: {accuracy * 100:.2f}%")
print(f"Improvement: {(accuracy - 0.712) * 100:+.2f} percentage points")
