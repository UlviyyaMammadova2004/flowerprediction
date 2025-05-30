import joblib
import json
import numpy as np

def init():
    global model
    model_path = "iris_model.pkl"
    model = joblib.load(model_path)

def run(data):
    try:
        inputs = json.loads(data)["data"]
        prediction = model.predict(np.array(inputs))
        return prediction.tolist()
    except Exception as e:
        return str(e)
