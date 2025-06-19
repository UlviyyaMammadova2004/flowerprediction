import joblib
import json
import numpy as np

def init():
    global model, scaler
    scaler, model = joblib.load("iris_model.pkl")

def run(data):
    try:
        inputs = json.loads(data)["data"]
        inputs_scaled = scaler.transform(np.array(inputs))
        prediction = model.predict(inputs_scaled)
        return prediction.tolist()
    except Exception as e:
        return str(e)

