from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("iris_model.pkl")

@app.route("/")
def home():
    return "✅ Iris Model API is running on Render!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        prediction = model.predict(np.array(data["data"]))
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
