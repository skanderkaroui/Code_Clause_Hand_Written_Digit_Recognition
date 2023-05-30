import tensorflow as tf
from PIL import Image
import numpy as np
import os, io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import torch
import config




interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def model_predict(img):
    input_data = img.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data.argmax(), output_data[0][output_data.argmax()]

def preprocess_image(image, img_type="file"):
    if img_type == "file":
        return (image.reshape(1, 28, 28, 1) / 255.).astype(np.float32)
    elif img_type == "df_row":
        return (image.to_numpy().reshape(1, 28, 28, 1) / 255.).astype(np.float32)

def predict(image_data):
    # Import image
    image = Image.open(io.BytesIO(image_data))

    # Convert the RGB image to grayscale image
    image = image.convert("L")

    # Resize the image to 28x28
    image = image.resize((28, 28))

    # Convert the image into numpy array
    image = np.array(image)

    # Reshape the image for the model
    image = image.reshape(1, 28, 28, 1) 

    # Normalize the pixel values in image
    image = image / 255.

    # Set the datatype of image as float32
    image = image.astype(np.float32)

    # Make prediction on the image
    prediction, confidence = model_predict(image)
    return prediction, confidence

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(prediction, confidence):
    # Create a bar plot

    prediction = float(prediction)
    confidence = int(float(confidence)*100)
    fig, ax = plt.subplots()
    ax.bar([prediction], [confidence], align='center', width=0.5)  # Adjust the width of the bar

    # Set the title and labels
    ax.set_title('Prediction')
    ax.set_xlabel('Number')
    ax.set_ylabel('Confidence (%)')

    # Set the tick labels for x-axis
    ax.set_xticks([prediction])
    ax.set_xticklabels([prediction])

    # Add text annotation for the bar
    ax.text(prediction, confidence + 1, str(confidence), ha='center')

    # Save the plot as an image
    plot_path = 'static/plot.png'
    plt.savefig(plot_path)

    return plot_path




'''
MODEL = None
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


def register_hook():
    save_output = SaveOutput()
    hook_handles = []

    for layer in MODEL.modules():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)
    return save_output


def module_output_to_numpy(tensor):
    return tensor.detach().to('cpu').numpy()


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{0:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def prob_img(probs):
    fig, ax = plt.subplots()
    rects = ax.bar(range(len(probs)), probs)
    ax.set_xticks(range(len(probs)), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    ax.set_ylim(0, 110)
    ax.set_title('Probability % of Digit by Model')
    autolabel(rects, ax)
    probimg = BytesIO()
    fig.savefig(probimg, format='png')
    probencoded = base64.b64encode(probimg.getvalue()).decode('utf-8')
    return probencoded


def interpretability_img(save_output):
    images = module_output_to_numpy(save_output.outputs[0])
    with plt.style.context("seaborn-white"):
        fig, _ = plt.subplots(figsize=(20, 20))
        plt.suptitle("Interpretability by Model", fontsize=50)
        for idx in range(16):
            plt.subplot(4, 4, idx+1)
            plt.imshow(images[0, idx])
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    interpretimg = BytesIO()
    fig.savefig(interpretimg, format='png')
    interpretencoded = base64.b64encode(
        interpretimg.getvalue()).decode('utf-8')
    return interpretencoded

def mnist_prediction(image_data):
    save_output = register_hook()
    img = img.to(DEVICE, dtype=torch.float)
    outputs = MODEL(x=image_data)

    probs = torch.exp(outputs.data)[0] * 100
    probencoded = prob_img(probs)
    interpretencoded = interpretability_img(save_output)

    _, output = torch.max(outputs.data, 1)
    pred = module_output_to_numpy(output)
    return probencoded, interpretencoded

'''