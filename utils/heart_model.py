from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
import os


current_dir = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = os.path.join(current_dir, '../pickle_files', 'heart_model_pickle.pkl') 

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)



feature_order = [
    'WeightInKilograms',
    'BMI',
    'HadAngina',
    'HadCOPD',
    'HadKidneyDisease',
    'DifficultyConcentrating',
    'DifficultyWalking',
    'ChestScan',
    'AlcoholDrinkers',
    'CovidPos',
    'HeighInFeet', 
    'GeneralHealth_Excellent',
    'GeneralHealth_Fair',
    'GeneralHealth_Good',
    'GeneralHealth_Poor',
    'GeneralHealth_Very good',
    'Sex_Female',
    'Sex_Male',
    'SmokerStatus_Former smoker',
    'SmokerStatus_Never smoked',
    'SmokerStatus_smokes',
    'AgeCategory_0-9',
    'AgeCategory_10-19',
    'AgeCategory_20-24',
    'AgeCategory_25-59',
    'AgeCategory_60 or older'
]

COLUMNS = [
    'WeightInKilograms', 'BMI', 'HadAngina', 'HadCOPD', 'HadKidneyDisease',
    'DifficultyConcentrating', 'DifficultyWalking', 'ChestScan',
    'AlcoholDrinkers', 'CovidPos', 'HeighInFeet', 'GeneralHealth_Excellent',
    'GeneralHealth_Fair', 'GeneralHealth_Good', 'GeneralHealth_Poor',
    'GeneralHealth_Very good', 'Sex_Female', 'Sex_Male',
    'SmokerStatus_Former smoker', 'SmokerStatus_Never smoked',
    'SmokerStatus_smokes', 'AgeCategory_0-9', 'AgeCategory_10-19',
    'AgeCategory_20-24', 'AgeCategory_25-59', 'AgeCategory_60 or older'
]


SCALES = {
    'WeightInKilograms': (30, 200),  
    'BMI': (10, 50),                 
    'HeighInFeet': (4, 7)            
}


GENERAL_HEALTH = ['Excellent', 'Fair', 'Good', 'Poor', 'Very good']
SEX = ['Female', 'Male']
SMOKER_STATUS = ['Former smoker', 'Never smoked', 'smokes']
AGE_CATEGORY = ['0-9', '10-19', '20-24', '25-59', '60 or older']

def scale_value(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def predict_disease_heart(data):
    print(data)
    try:
        features = []

       
        for col in ['WeightInKilograms', 'BMI', 'HeighInFeet']:
            if col not in data:
                raise ValueError(f"Missing value for feature: {col}")
            min_val, max_val = SCALES[col]
            scaled_value = scale_value(float(data[col]), min_val, max_val)
            features.append(scaled_value)

        
        for col in ['HadAngina', 'HadCOPD', 'HadKidneyDisease',
                    'DifficultyConcentrating', 'DifficultyWalking', 'ChestScan',
                    'AlcoholDrinkers', 'CovidPos']:
            if col not in data:
                raise ValueError(f"Missing value for feature: {col}")
            if data[col]=="Yes":
                features.append(1)
            else:
                features.append(0)
        

        
        general_health_onehot = [1 if data.get('GeneralHealth') == category else 0 for category in GENERAL_HEALTH]
        features.extend(general_health_onehot)

        sex_onehot = [1 if data.get('gender') == category else 0 for category in SEX]
        features.extend(sex_onehot)

        smoker_status_onehot = [1 if data.get('SmokerStatus') == category else 0 for category in SMOKER_STATUS]
        features.extend(smoker_status_onehot)

        age_category_onehot = [1 if data.get('AgeCategory') == category else 0 for category in AGE_CATEGORY]
        features.extend(age_category_onehot)

        
        if len(features) != len(COLUMNS):
            return {'error': f'Input data shape mismatch. Expected {len(COLUMNS)} features, got {len(features)}'}

        
        features = np.array(features).reshape(1, -1)

        
        output = model.predict(features)
        probability = float(output[0])  
        prediction = int(probability > 0.5)

        return {
            'prediction': prediction,
            'probability': probability
        }

    except Exception as e:
        return {'error': str(e)}
