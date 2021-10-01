import os
import csv
import shutil
import pandas as pd
import numpy as np 
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file
from flask_cors import CORS, cross_origin
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = Flask(__name__)


app.config["sample_file"] = "Prediction_SampleFile/"


@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/return_sample_file/')
@cross_origin()
def return_sample_file():
    sample_file = os.listdir("Prediction_SampleFile/")[0]
    return send_from_directory(app.config["sample_file"], sample_file)




def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    
    if request.method == 'POST':
        
        try:
            
            if 'imagefile' not in request.files:
                return render_template("invalid.html")
            
            file = request.files['imagefile']
            
            MODEL = tf.keras.models.load_model("..\\Potato-Disease\\saved_models\\1")

            CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

            path = 'Prediction_InputFileFromUser/'
            
            image = read_file_as_image(file.read())
            
            img_batch = np.expand_dims(image, 0)
    
            predictions = MODEL.predict(img_batch)
    
            index = np.argmax(predictions[0])
    
            predicted_class = CLASS_NAMES[index]
    
            confidence = np.max(predictions[0])
    
            #return {"class": predicted_class, 
                   
                   #"confidence": float(confidence)}
                   
            result =  {"Predicted Class for Given Image:": predicted_class, 
                   
                   "Confidence of Model:": round(float(confidence),2)}      
            
            #print(predicted_class,confidence)
            
            return render_template("result.html",result = result)
        
            
        except Exception as e:
            return render_template("invalid.html")



if __name__ == '__main__':
    app.run(debug=True)