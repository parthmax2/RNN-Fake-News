import numpy as np
import tensorflow as tf
import pickle
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load trained RNN model
model = tf.keras.models.load_model("models/rnn_model.keras")

# Load tokenizer
with open("models/tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

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
    data = request.form["news_text"]
    processed_text = preprocess_text(data)
    
    prediction_prob = model.predict(processed_text)[0][0]  # Get single prediction
    result = "Fake News" if prediction_prob > 0.5 else "Real News"
    
    return jsonify({"prediction": result, "confidence": float(prediction_prob)})

if __name__ == "__main__":
    app.run(debug=True)
