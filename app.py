from flask import Flask, request, render_template, jsonify
import base64
from model import predict, plot

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict-digit", methods=["POST"])
def predict_digit():
    image = request.get_json(silent=True)['image'].split(",")[1]
    image_data = base64.b64decode(image)
    prediction, confidence = predict(image_data)
    plot_data = plot(prediction, confidence)  # Generate the plot and get the base64-encoded plot data
    response = { 
        "prediction": str(prediction),
        "confidence": str(confidence),
        "plot_data": plot_data
    }
    return jsonify(response)