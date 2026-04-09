import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory

model = tf.keras.models.load_model(
    os.path.join(os.path.dirname(__file__), "../stock_dl_model.keras")
)

app = Flask(__name__)

@app.route("/")
def index():
    return send_from_directory(
        os.path.join(os.path.dirname(__file__), "../stock_fixed/templates"),
        "index.html"
    )

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        inputs = np.array([data["inputs"]], dtype=float)
        prediction = model.predict(inputs)
        return jsonify({"prediction": prediction[0].tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500