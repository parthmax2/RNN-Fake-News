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

# Load trained RNN model
try:
    model = tf.keras.models.load_model("models/rnn_model.keras")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Load tokenizer
try:
    with open("models/tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    logging.info("Tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading tokenizer: {e}")
    tokenizer = None

# Max length for padding (same as used in training)
MAX_LENGTH = 500  

def preprocess_text(text):
    """Tokenizes and pads input text for the RNN model"""
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
    
    data = request.form.get("news_text", "").strip()
    
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