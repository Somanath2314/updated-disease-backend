from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, '../pickle_files', 'stroke_model_pickle.pkl') 

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)


app = Flask(__name__)
CORS(app)

# COLUMNS = [
#     "gender",
#     "hypertension",
#     "heart_disease",
#     "ever_married",
#     "Residence_type",
#     "avg_glucose_level",
#     "bmi",
#     "AgeCategory_0-9",
#     "AgeCategory_10-19",
#     "AgeCategory_20-24",
#     "AgeCategory_25-59",
#     "AgeCategory_60 or older",
#     "Never smoked",
#     "smokes",
#     "Former smoker",
#     "work_type_private",
#     "work_type_self_employed",
#     "work_type_govt",
#     "work_type_children",
#     "work_type_unemployed"
# ]



@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API!"

def predict_disease_stroke(data):
    try:
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        features = []

        # Gender
        features.append(1 if data.get("Gender") == "Male" else 0)

        # Hypertension
        features.append(1 if data.get("Hypertension") == "Yes" else 0)

        # Heart Disease
        features.append(1 if data.get("HeartDisease") == "Yes" else 0)

        # Ever Married
        features.append(1 if data.get("EverMarried") == "Yes" else 0)

        # Residence Type
        features.append(1 if data.get("ResidenceType") == "Urban" else 0)

        # Average Glucose Level
        features.append(float(data.get("blood_glucose_level")))

        # BMI
        features.append(float(data.get("BMI")))

        # Age Category (One-Hot Encoding)
        age_category = data.get("AgeCategory")
        age_categories = ["0-9", "10-19", "20-24", "25-59", "60 or older"]
        for category in age_categories:
            features.append(1 if age_category == category else 0)

        # Smoking Status (One-Hot Encoding)
        smoking_status = data.get("SmokerStatus")
        smoking_categories = ["Never smoked", "smokes", "Former smoker"]
        for category in smoking_categories:
            features.append(1 if smoking_status == category else 0)

        # Work Type (One-Hot Encoding)
        work_type = data.get("WorkType")
        work_categories = [
            "Private",
            "Self-employed",
            "Govt_job",
            "children",
            "Never_worked",
        ]
        for category in work_categories:
            features.append(1 if work_type == category else 0)

        # Debugging: Verify feature vector length and values
        print(f"Features: {features}, Length: {len(features)}")

        # Validate feature length
        if len(features) != 20:
            return jsonify({"error": f"Feature mismatch: expected 20, got {len(features)}"}), 400

        # Prediction
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return {"prediction": int(prediction), "probability": float(probability)}

    except Exception as e:
        return {"error": str(e)}, 500



# @app.route('/strokepredict', methods=['POST'])
# def stroke_predict():
#     try:
#         # Get JSON data from the request
#         data = request.get_json()

#         # Call the prediction function
#         result = predict_disease_stroke(data)

#         # Return the result as a Response object
#         return result

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500





# if __name__ == "__main__":
#     app.run(debug=True)

