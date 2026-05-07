from flask import Flask, render_template, request
import pickle
import numpy as np
history = []

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["T"]),
        float(request.form["TM"]),
        float(request.form["Tm"]),
        float(request.form["SLP"]),
        float(request.form["H"]),
        float(request.form["VV"]),
        float(request.form["V"]),
        float(request.form["VM"])
    ]

    prediction = round(model.predict([features])[0], 2)

    # 🔥 history me add karo
    history.append(prediction)

    # sirf last 5 values rakho
    if len(history) > 5:
        history.pop(0)

    return render_template("index.html",
                           prediction=prediction,
                           history=history)

if __name__ == "__main__":
    app.run(debug=True)