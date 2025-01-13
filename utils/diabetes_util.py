from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, '../pickle_files', 'diabetes_model_pickle.pkl') 

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)


app = Flask(__name__)
CORS(app)

@app.route('/soma')
def home():
    return "Welcome to the Diabetes Prediction API!"



# cols on which the model is trained, diabetes is the y variable
# COLUMNS = [
#     "gender", 
#     "hypertension", 
#     "heart_disease", 
#     "bmi", 
#     "HbA1c_level", 
#     "blood_glucose_level",  
#     "AgeCategory_0-9", 
#     "AgeCategory_10-19", 
#     "AgeCategory_20-24", 
#     "AgeCategory_25-59", 
#     "AgeCategory_60 or older", 
#     "Never smoked", 
#     "smokes", 
#     "Former smoker"
# ]


# @app.route('/')
# def hello():
#     return "Welcome to the Diabetes Prediction API!" 
def predict_disease_diabetes(data):
    try: 

        # Check if data is provided
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        features = []

        # Gender
        features.append(1 if data.get("Gender") == "Male" else 0)

        # Hypertension
        features.append(1 if data.get("Hypertension") == "Yes" else 0)

        # Heart Disease
        features.append(1 if data.get("HeartDisease") == "Yes" else 0)

        # BMI
        features.append(float(data.get("BMI")))

        # HbA1c Level
        features.append(float(data.get("HbA1c_level")))

        # Blood Glucose Level
        features.append(int(data.get("blood_glucose_level")))

        # Age Category (One-Hot Encoding)
        age_category = data.get("AgeCategory")
        age_categories = ["0-9", "10-19", "20-24", "25-59", "60 or older"]
        for cat in age_categories:
            features.append(1 if age_category == cat else 0)

        # Smoker Status (One-Hot Encoding)
        smoker_status = data.get("SmokerStatus")
        smoker_categories = ["Never smoked", "smokes", "Former smoker"]
        for status in smoker_categories:
            features.append(1 if smoker_status == status else 0)

        # Convert to NumPy array and make prediction
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]  # Assuming binary classification
        probability = model.predict_proba(features)[0][1]
        return {"prediction": int(prediction), "probability": float(probability)}

    except Exception as e:
        return {"error": str(e)}, 500

# @app.route('/diabetespredict', methods=['POST'])
# def diabetes_predict():
#     try:
#         # Get JSON data from the request
#         data = request.get_json()

#         # Call the prediction function
#         result = predict_disease_diabetes(data)

#         # Return the result as a Response object
#         return result

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)