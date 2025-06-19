# main.py
from flask import Flask, request, render_template
import numpy as np
import joblib
import os
from database import collection  # MongoDB bağlantısı

app = Flask(__name__)
scaler, model = joblib.load("iris_model.pkl")  # scaler ve model birlikte yüklenir

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict-form", methods=["GET", "POST"])
def predict_form():
    if request.method == "POST":
        try:
            # Girdi al
            f1 = float(request.form["f1"])
            f2 = float(request.form["f2"])
            f3 = float(request.form["f3"])
            f4 = float(request.form["f4"])

            # Ölçekleme ve tahmin
            input_data = np.array([[f1, f2, f3, f4]])
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)[0]

            # MongoDB'ye kaydet
            collection.insert_one({
                "sepal_length": f1,
                "sepal_width": f2,
                "petal_length": f3,
                "petal_width": f4,
                "predicted_class": int(prediction)
            })

            return render_template("form.html", prediction=prediction, f1=f1, f2=f2, f3=f3, f4=f4)

        except Exception as e:
            return render_template("form.html", error=str(e))

    return render_template("form.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

