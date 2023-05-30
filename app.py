from flask import Flask, request, render_template, jsonify
import base64, os, random
from uuid import uuid4
from model import predict
from model import plot
import random
# from model import mnist_prediction

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict-digit", methods=["POST", "GET"])
def predict_digit():
    image = request.get_json(silent=True)['image'].split(",")[1]
    image_data = base64.urlsafe_b64decode(image)

    prediction, confidence = predict(image_data)
   
    random_param = random.randint(1, 100000)

    plot_path = plot(prediction, confidence)  # Generate the plot and get the plot path


    response = { 
        "prediction": str(prediction),
        "confidence": str(confidence),
        "plot_path": plot_path,
        "random":random_param
    }

    return jsonify(response)