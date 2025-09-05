from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")

scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    features = np.array(request.json["features"]).reshape(1, -1)
    scaled = scaler.transform(features)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][pred]
    return jsonify({"prediction": int(pred), "confidence": round(prob, 2)})

@app.route("/")
def home():
    return "API is running!"

if __name__ == "__main__":
    app.run(debug=True)
