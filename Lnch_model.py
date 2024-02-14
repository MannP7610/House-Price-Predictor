# app.py

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.externals import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('your_model.pkl')  # Replace 'your_model.pkl' with the path to your trained model file

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = []
        # Retrieve feature values from the form
        # Example: features.append(float(request.form['feature1']))
        # Repeat this for all features
        
        # Convert features to a numpy array
        features_array = np.array([features])

        # Use the loaded model to make predictions
        prediction = model.predict(features_array)

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
