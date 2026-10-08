import os

import firebase_admin
from firebase_admin import credentials, firestore

from flask import Flask, request, jsonify
import torch
import joblib
from transformers import BertTokenizerFast, BertForSequenceClassification
from huggingface_hub import hf_hub_download

#my Flask application, which provides the API endpoints such as /predict and /health
app = Flask(__name__)

# ============================================================
# Firebase Firestore
# ============================================================

# cred = credentials.Certificate("firebase-service-account.json")

# firebase_admin.initialize_app(cred)

# db = firestore.client()
# ============================================================
# Firebase Firestore
# ============================================================

FIREBASE_CREDENTIALS_PATH = os.getenv(
    "FIREBASE_CREDENTIALS_PATH",
    "firebase-service-account.json"
)

cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)

firebase_admin.initialize_app(cred)

db = firestore.client()

# ============================================================
# Hugging Face model
# ============================================================

#loading my emotion model

MODEL_PATH = "Hasandi/nutrimind-emotion-model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Load model once
# ============================================================

# Loaded once at import time.
# Under gunicorn we run a single worker so exactly
# one model copy lives in memory.

tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)

model.eval()


# ============================================================
# Load label encoder
# ============================================================

label_encoder_path = hf_hub_download(
    repo_id=MODEL_PATH,
    filename="label_encoder.pkl"
)

le = joblib.load(label_encoder_path)

labels = le.classes_

MODEL_LOADED = True


print("Model loaded successfully.")
print("Emotion classes:", list(labels))
print("Device:", device)


# ============================================================
# Emotion Prediction
# ============================================================

def predict_emotion(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Move input tensors to the same device as the model
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    #with torch.no_grad():

        #outputs = model(**inputs)

        #probabilities = torch.softmax(
           # outputs.logits,
           # dim=1
       # )

    with torch.inference_mode():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )



        # Get predicted emotion
        pred = torch.argmax(
            probabilities,
            dim=1
        ).item()

        # Get confidence of predicted emotion
        confidence = probabilities[0][pred].item()

    #emotion = str(labels[pred])

    #return emotion, float(confidence)

        # Create complete probability distribution
        probability_distribution = {
            str(labels[index]): round(
                probabilities[0][index].item(),
                4
            )
            for index in range(len(labels))
        }

    emotion = str(labels[pred])

    return emotion, float(confidence), probability_distribution


# ============================================================
# Emotion Suggestions from Firestore
# ============================================================

def get_suggestion(emotion, confidence):

    # Determine which confidence range to use
    if confidence >= 0.90:
        confidence_field = "confidence_90"

    elif confidence >= 0.75:
        confidence_field = "confidence_75"

    elif confidence >= 0.50:
        confidence_field = "confidence_50"

    elif confidence >= 0.25:
        confidence_field = "confidence_25"

    else:
        confidence_field = "confidence_low"


    # Convert emotion to lowercase so it matches
    # the Firestore document ID
    emotion = emotion.strip().lower()


    # Get the emotion document from Firestore
    doc_ref = db.collection("suggestions").document(emotion)

    doc = doc_ref.get()


    # Check whether the document exists
    if not doc.exists:

        return (
            "Take a moment to check in with yourself.\n"
            "Give yourself some time to relax and reflect.\n"
            "Be gentle with yourself and your emotions."
        )


    data = doc.to_dict()


    # Get the three suggestions from the selected
    # confidence field
    suggestions = data.get(confidence_field, [])


    # Check whether suggestions exist
    if not suggestions:

        return (
            "Take a moment to check in with yourself.\n"
            "Give yourself some time to relax and reflect.\n"
            "Be gentle with yourself and your emotions."
        )


    # Display all three suggestions
    formatted_suggestions = []

    for index, suggestion in enumerate(suggestions, start=1):

        formatted_suggestions.append(
            f"{index}. {suggestion}"
        )


    return "\n".join(formatted_suggestions)


# ============================================================
# Home
# ============================================================

@app.route("/")
def home():

    return "NutriMind Emotion API is running successfully!"


# ============================================================
# Health Check
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED
    })


# ============================================================
# Emotion Prediction API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True)


    # Check whether text was provided
    if not data or "text" not in data:

        return jsonify({
            "error": "Text is required"
        }), 400


    text = data["text"]


    # Check that text is actually a string
    if not isinstance(text, str) or not text.strip():

        return jsonify({
            "error": "Text must be a non-empty string"
        }), 400


    # --------------------------------------------------------
    # Predict emotion
    # --------------------------------------------------------

    #emotion, confidence = predict_emotion(text)
    emotion, confidence, probability_distribution = predict_emotion(text)


    # --------------------------------------------------------
    # Generate suggestion
    # --------------------------------------------------------

    suggestion = get_suggestion(
        emotion,
        confidence
    )


    # --------------------------------------------------------
    # Return complete result
    # --------------------------------------------------------

    return jsonify({

        "text": text,

        "emotion": emotion,

        "confidence": round(confidence, 4),

        "probabilities": probability_distribution,

        "suggestion": suggestion

    })


# ============================================================
# Run Flask application
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
