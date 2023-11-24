# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)

# Load the trained models
rf_model = joblib.load("rf_model.pkl")
lgbm_model = joblib.load("lgbm_model.pkl")

# Your preprocessing functions go here

# Your feature extraction and data preprocessing code go here

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from the form
        # Extract and preprocess features
        # Make predictions using the loaded models

        # Example (modify according to your data)
        input_data = request.form['input_data']
        #processed_data = preprocess_data(input_data)
        
        # Make predictions using the models
        #rf_prediction = rf_model.predict(processed_data)
        l#gbm_prediction = lgbm_model.predict(processed_data)

        return render_template('index.html', rf_prediction=rf_prediction, lgbm_prediction=lgbm_prediction)

if __name__ == '__main__':
    app.run(debug=True)
