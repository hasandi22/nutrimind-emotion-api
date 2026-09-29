import os

from flask import Flask, request, jsonify
import torch
import joblib
from transformers import BertTokenizerFast, BertForSequenceClassification
from huggingface_hub import hf_hub_download

app = Flask(__name__)


# ============================================================
# Hugging Face model
# ============================================================

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

    with torch.no_grad():

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

    emotion = str(labels[pred])

    return emotion, float(confidence)


# ============================================================
# Emotion Suggestions
# ============================================================

def get_suggestion(emotion, confidence):

    # If the model is not sufficiently confident,
    # give a general response instead of a specific suggestion.

    if confidence < 0.60:

        return (
            "I'm not very sure about your emotion. "
            "Try expressing more details about how you feel."
        )


    suggestions = {

        "joy":
            "You're feeling happy 😊 Keep doing things that bring you joy "
            "and share your positivity with others.",

        "love":
            "You seem emotionally connected ❤️ Cherish your relationships "
            "and express gratitude to people you care about.",

        "positive":
            "You have a positive mindset 🌟 Maintain this energy by "
            "continuing healthy habits and self-care.",

        "surprise":
            "You seem surprised 😲 Take a moment to process what's "
            "happening before reacting.",

        "sadness":
            "You may be feeling sad 😔 It's okay to feel this way. "
            "Try talking to someone you trust or doing something comforting.",

        "fear":
            "You seem anxious or fearful 😟 Try deep breathing, "
            "grounding exercises, and remind yourself you're safe.",

        "anger":
            "You seem angry 😡 Take a break, step away from the situation, "
            "and try calming breathing techniques.",

        "stress":
            "You appear stressed 😥 Try organizing your thoughts, "
            "resting, or taking short breaks.",

        "disgust":
            "You may be feeling discomfort or dislike 🤢 "
            "Try to distance yourself from the trigger and reset your thoughts.",

        "neutral":
            "You are emotionally balanced 😐 Stay mindful and keep "
            "maintaining stability in your daily routine."
    }


    return suggestions.get(
        emotion,
        "Take care of yourself and stay mindful of your emotions."
    )


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

    emotion, confidence = predict_emotion(text)


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
```
