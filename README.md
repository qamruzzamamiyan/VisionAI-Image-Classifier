# VisionAI — AI Image Classifier

An AI-powered Cat vs Dog Image Classification application built using Python, PyTorch, CNN, and Flask.

## Features

- Cat and Dog Image Classification
- Custom CNN V1 and CNN V2 Models
- Real-time Image Prediction
- Prediction Confidence Scores
- V1 vs V2 Model Comparison
- Modern Web Dashboard
- Prediction History

## Model Performance

| Model | Test Accuracy |
|---|---|
| CNN V1 | 71.20% |
| CNN V2 | 82.40% |

**Improvement: +11.20 percentage points**

## Technologies

Python, PyTorch, Torchvision, Flask, Scikit-learn, HTML, CSS, JavaScript.

## Setup

1. Create a Python virtual environment.
2. Install dependencies using `pip install -r requirements.txt`.
3. Download the Cats and Dogs Filtered Dataset and extract it into the project directory.
4. Run `python3 prepare_data.py`.
5. Run `python3 train.py` and `python3 train_v2.py` to generate both model files.
6. Run `python3 app.py`.
7. Open http://127.0.0.1:5000 in your browser.

## Author

Qamruzzama Miyan  
B.Tech CSE (AI & ML)  
ABES Engineering College