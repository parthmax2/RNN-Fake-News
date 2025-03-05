import os
import numpy as np
import tensorflow as tf
import pickle
import logging
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Model & tokenizer paths
MODEL_PATH = "models/fake_news_rnn.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"
MAX_LENGTH = 500  # Max sequence length used during training

# Load trained RNN model
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        model = None
else:
    logging.error("Model file not found!")
    model = None

# Load tokenizer
if os.path.exists(TOKENIZER_PATH):
    try:
        with open(TOKENIZER_PATH, "rb") as handle:
            tokenizer = pickle.load(handle)
        logging.info("Tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        tokenizer = None
else:
    logging.error("Tokenizer file not found!")
    tokenizer = None

def preprocess_text(text):
    """Tokenizes and pads input text for the RNN model."""
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=MAX_LENGTH)
    return padded_sequence

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handles prediction requests"""
    if not model or not tokenizer:
        return jsonify({"error": "Model or tokenizer not loaded properly"}), 500

    data = request.json.get("news", "").strip()  # Fetch input from JSON body

    if not data:
        return jsonify({"error": "No input text provided"}), 400

    try:
        processed_text = preprocess_text(data)
        prediction_prob = model.predict(processed_text)[0][0]  # Get single prediction
        result = "Fake News" if prediction_prob > 0.5 else "Real News"

        return jsonify({
            "prediction": result,
            "confidence": round(float(prediction_prob), 4)
        })
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
