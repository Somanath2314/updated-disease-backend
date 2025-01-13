from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os

# Set the path for the model and scaler
MODEL_PATH = os.path.join(r'pickle_files', 'asthma_model.pkl')
SCALER_PATH = os.path.join(r'pickle_files', 'asthma_scaler.pkl')

# Load the model if it exists
if not os.path.exists(MODEL_PATH):
    print("Model file not found at:", MODEL_PATH)
else:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)

# Load the scaler if it exists
if not os.path.exists(SCALER_PATH):
    print("Scaler file not found at:", SCALER_PATH)
else:
    with open(SCALER_PATH, 'rb') as file:
        scaler = pickle.load(file)

# Define columns and scale ranges
COLUMNS = [
    'Gender', 'Ethnicity', 'BMI', 'Smoking', 'PhysicalActivity',
    'SleepQuality', 'PollutionExposure', 'PollenExposure', 'DustExposure',
    'PetAllergy', 'ShortnessOfBreath', 'ChestTightness', 'Coughing',
    'NighttimeSymptoms', 'ExerciseInduced'
]

# Scale ranges for continuous variables
SCALES = {
    'BMI': (15, 40),                   # Body Mass Index (15 to 40)
    'PhysicalActivity': (0, 10),       # Weekly Physical Activity (0 to 10 hours)
    'SleepQuality': (4, 10),           # Sleep Quality Score (4 to 10)
    'PollutionExposure': (0, 10),      # Pollution Exposure Score (0 to 10)
    'PollenExposure': (0, 10),         # Pollen Exposure Score (0 to 10)
    'DustExposure': (0, 10),
    'Ethnicity':(0,3),                 # Ethnicity scaled value
}

# Categorical variables
ETHNICITY = ["Caucasian", "African American", "Asian", "Other"]
SEX = ['Male', 'Female']
SMOKER_STATUS = ['Former smoker', 'Never smoked', 'smokes']

# Function to scale values
def scale_value(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)



# Function to process and predict asthma with probability
def predict_asthma(data):
    try:
        features = []

        # Convert string values to appropriate types
        try:
            bmi = float(data.get('BMI', 0))
        except (ValueError, TypeError):
            # Calculate BMI if direct value is invalid
            weight = float(data.get('WeightInKilograms', 0))
            height = float(data.get('HeightInMeters', 1))  # Default to 1 to avoid division by zero
            bmi = weight / (height * height) if height > 0 else 0

        # One-hot encoding for Gender
        sex = 0 if data.get('Gender', '').lower() == 'male' else 1
        features.append(sex)

        # Label encoding for Ethnicity
        ethnicity_mapping = {
            'caucasian': 0,
            'african american': 1,
            'asian': 2,
            'other': 3
        }
        ethnicity = data.get('Ethnicity', '').lower()
        ethnicity_index = ethnicity_mapping.get(ethnicity, 3)  # Default to 'Other' if not found
        features.append(ethnicity_index)

        # BMI scaling
        min_val, max_val = SCALES['BMI']
        scaled_bmi = max(min(bmi, max_val), min_val)  # Clamp values to valid range
        scaled_value = scale_value(scaled_bmi, min_val, max_val)
        features.append(scaled_value)

        # Smoker Status encoding
        smoker_status = data.get('SmokerStatus', '').lower()
        smoker_index = 1 if smoker_status in ['former smoker', 'smokes'] else 0
        features.append(smoker_index)

        # Handle continuous features with default values and scaling
        continuous_features = {
            'PhysicalActivity': data.get('PhysicalActivity', 0),
            'SleepQuality': data.get('SleepHours', 7),  # Using SleepHours as SleepQuality
            'PollutionExposure': 5,  # Default middle value
            'PollenExposure': 5,     # Default middle value
            'DustExposure': 5        # Default middle value
        }

        for feature, value in continuous_features.items():
            try:
                value = float(value)
                min_val, max_val = SCALES[feature]
                scaled_value = scale_value(max(min(value, max_val), min_val), min_val, max_val)
                features.append(scaled_value)
            except (ValueError, TypeError):
                return {"error": f"Invalid value for {feature}"}, 400

        # Map symptoms from input data to binary values
        symptom_mapping = {
            'PetAllergy': 'No',  # Default value since not in input
            'ShortnessOfBreath': data.get('Breathlessness', 'No'),
            'ChestTightness': data.get('ChestPain', 'No'),
            'Coughing': 'Yes' if data.get('HadCOPD', 'No') == 'Yes' else 'No',
            'NighttimeSymptoms': 'Yes' if data.get('SleepHours', 8) < 6 else 'No',
            'ExerciseInduced': data.get('DifficultyWalking', 'No')
        }

        # Add symptom features (binary)
        for symptom in symptom_mapping.values():
            value = 1 if symptom == 'Yes' else 0
            features.append(value)

        # Feature count validation
        if len(features) != len(COLUMNS):
            return {'error': f'Input data shape mismatch. Expected {len(COLUMNS)} features, got {len(features)}'}, 400

        # Convert features to a numpy array
        features = np.array(features).reshape(1, -1)

        # Predict asthma using the model
        output = model.predict(features)
        prediction_prob = model.predict_proba(features)[0][1]

        return {
            'prediction': int(output[0] > 0.5),
            'probability': float(prediction_prob)
        }

    except Exception as e:
        return {"error": str(e)}, 500