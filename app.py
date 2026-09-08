from flask import Flask, request, jsonify
import torch
import joblib
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Hugging Face model
MODEL_PATH = "Hasandi/nutrimind-emotion-model"

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Use CPU or GPU automatically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

# Load label encoder from Hugging Face
from huggingface_hub import hf_hub_download

label_encoder_path = hf_hub_download(
    repo_id=MODEL_PATH,
    filename="label_encoder.pkl"
)

le = joblib.load(label_encoder_path)
labels = le.classes_


def predict_emotion(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        pred = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[0][pred].item()

    return labels[pred], confidence


@app.route("/")
def home():

    return "NutriMind Emotion API is running successfully!"


@app.route("/health")
def health():

    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data or "text" not in data:

        return jsonify({
            "error": "Text is required"
        }), 400

    text = data["text"]

    emotion, confidence = predict_emotion(text)

    return jsonify({
        "text": text,
        "emotion": emotion,
        "confidence": confidence
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )