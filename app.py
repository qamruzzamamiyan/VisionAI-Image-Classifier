
import io

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from flask import Flask, request, jsonify, render_template

from model import CatDogCNN
from model_v2 import CatDogCNNV2

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

device = torch.device("cpu")

# ---------- Load V1 ----------
v1_checkpoint = torch.load(
    "saved_models/cat_dog_model.pth",
    map_location=device,
    weights_only=True
)

model_v1 = CatDogCNN().to(device)
model_v1.load_state_dict(v1_checkpoint["model_state"])
model_v1.eval()

# ---------- Load V2 ----------
v2_checkpoint = torch.load(
    "saved_models/cat_dog_v2.pth",
    map_location=device,
    weights_only=True
)

model_v2 = CatDogCNNV2().to(device)
model_v2.load_state_dict(v2_checkpoint["model_state"])
model_v2.eval()

# ---------- Preprocessing ----------
transform_v1 = transforms.Compose([
    transforms.Resize(
        (v1_checkpoint["image_size"],
         v1_checkpoint["image_size"])
    ),
    transforms.ToTensor()
])

transform_v2 = transforms.Compose([
    transforms.Resize(
        (v2_checkpoint["image_size"],
         v2_checkpoint["image_size"])
    ),
    transforms.ToTensor()
])


def read_image():
    if "image" not in request.files:
        raise ValueError("No image uploaded")

    file = request.files["image"]
    data = file.read()

    if not data:
        raise ValueError("Empty image")

    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise ValueError("Invalid image")


def run_prediction(model, checkpoint, transform, image):
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    idx_to_class = {
        index: name
        for name, index in checkpoint["class_to_idx"].items()
    }

    prediction_index = probabilities.argmax().item()

    class_probabilities = {
        idx_to_class[i]: round(
            probabilities[i].item() * 100, 2
        )
        for i in range(len(idx_to_class))
    }

    return {
        "prediction": idx_to_class[prediction_index],
        "confidence": round(
            probabilities[prediction_index].item() * 100, 2
        ),
        "probabilities": class_probabilities
    }


@app.route("/")
def home():
    return render_template("index.html")


# Existing dashboard API - V2
@app.route("/predict", methods=["POST"])
def predict():
    try:
        image = read_image()
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    result = run_prediction(
        model_v2, v2_checkpoint, transform_v2, image
    )

    return jsonify({
        **result,
        "model_version": "CNN V2",
        "test_accuracy": 82.40
    })


# New V1 vs V2 comparison API
@app.route("/compare", methods=["POST"])
def compare():
    try:
        image = read_image()
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    v1_result = run_prediction(
        model_v1, v1_checkpoint, transform_v1, image
    )

    v2_result = run_prediction(
        model_v2, v2_checkpoint, transform_v2, image
    )

    return jsonify({
        "v1": {
            **v1_result,
            "test_accuracy": 71.20
        },
        "v2": {
            **v2_result,
            "test_accuracy": 82.40
        },
        "accuracy_improvement": 11.20
    })


if __name__ == "__main__":
    app.run(debug=True)
