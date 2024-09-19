
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('always')


app = Flask(__name__)

# Load the RandomForest model
RF_model_path = 'RandomForest.pkl'
with open(RF_model_path, 'rb') as file:
    RF_model = pickle.load(file)

# Load datasets
data1 = pd.read_csv("data/Crop_recommendation.csv")
data2 = pd.read_csv("data/csv")
data_soil = pd.read_csv("data/soil.csv")

# Create dictionaries for commodities and soil types
commodity_prices = dict(zip(data2['commodity'], data2['modal_price']))
soil_types = dict(zip(data_soil['State'], data_soil['Soil_type']))

@app.route('/crop_prediction', methods=['POST'])
def predict_crop():
    # Get JSON data from request
    data = request.json
    try:
        # Extract features from JSON
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        geolocation = data['geolocation'].capitalize()

        # Prepare data for prediction
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Make prediction using the RandomForest model
        prediction = RF_model.predict(input_data)
        crop_prediction = prediction[0].capitalize()

        # Determine the crop and profit
        crop_info = f"{crop_prediction} is not in the dataset."
        if crop_prediction in commodity_prices:
            profit = commodity_prices[crop_prediction]
            crop_info = f"{crop_prediction} is the BEST CROP to grow, and profit gained is RS {profit}"

        # Determine soil type for geolocation
        soil_info = f"Soil information for {geolocation} is not available."
        if geolocation in soil_types:
            soil_info = f"{geolocation} is majorly covered by {soil_types[geolocation]} soil."

        # Return results as JSON
        return jsonify({
            'recommended_crop': crop_prediction,
            'crop_info': crop_info,
            'soil_info': soil_info
        })
    
    except KeyError as e:
        return jsonify({"error": f"Missing input field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)