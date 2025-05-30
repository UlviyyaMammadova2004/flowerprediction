from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("iris_model.pkl")

@app.route("/")
def home():
    return "✅ Iris Model API is running on Render!"

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    prediction = model.predict(np.array(data["data"]))
    return jsonify({"prediction": prediction.tolist()})

# 🔸 HTML form gösterme
@app.route("/form")
def form():
    return render_template("form.html")

# 🔸 HTML üzerinden gelen formu işleyip sonucu gösterme
@app.route("/predict-form", methods=["POST"])
def predict_form():
    try:
        f1 = float(request.form["f1"])
        f2 = float(request.form["f2"])
        f3 = float(request.form["f3"])
        f4 = float(request.form["f4"])
        data = np.array([[f1, f2, f3, f4]])
        prediction = model.predict(data)[0]
        return render_template("form.html", prediction=prediction)
    except Exception as e:
        return render_template("form.html", prediction="Error: " + str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
