
from flask import Flask, request, jsonify
from flask_cors import CORS 
from utils.Asthma_model import predict_asthma  # Import the function from utils
from utils.heart_model import predict_disease_heart  # Import the function from utils
from utils.diabetes_util import predict_disease_diabetes
from utils.stroke_util import predict_disease_stroke 

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",  # Vite dev server
            "http://localhost:3000",
            "https://disease-prediction-app.vercel.app"  # Replace with your actual frontend domain,
                "*",
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Range", "X-Content-Range"],
        "supports_credentials": True,
        "max_age": 600
    }
})
@app.after_request
def after_request(response):
    allowed_origins = ['http://localhost:5173', 
                      'https://disease-prediction-app.vercel.app']
    origin = request.headers.get('Origin')
    if origin in allowed_origins:
        response.headers.add('Access-Control-Allow-Origin', origin)
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

@app.route('/')
def home():
    return "Welcome to the Disease Prediction API!"

@app.route('/heartpredict', methods=['POST'])
def predict_heart_route():
    try:
        # Parse JSON input
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid input data'}), 400

        # Call prediction function
        heart_result = predict_disease_heart(data)

        # Check for errors in the result
        if 'error' in heart_result:
            return jsonify(heart_result), 400

        return jsonify(heart_result)
    except Exception as e:
        app.logger.error(f"Error in /heartpredict: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/asthmapredict', methods=['POST'])
def asthama_predict():
    try:
        # Parse JSON input
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid input data'}), 400

        # Call prediction function
        asthma_result = predict_asthma(data)

        # Check for errors in the result
        if 'error' in asthma_result:
            return jsonify(asthma_result), 400

        return jsonify(asthma_result)
    except Exception as e:
        app.logger.error(f"Error in /asthmapredict: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/diabetespredict', methods=['POST'])
def diabetes_predict():
    try:
        # Parse JSON input
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid input data'}), 400

        # Call prediction function
        diabetes_result = predict_disease_diabetes(data)

        # Check for errors in the result
        if 'error' in diabetes_result:
            return jsonify(diabetes_result), 400

        return jsonify(diabetes_result)
    except Exception as e:
        app.logger.error(f"Error in /diabetespredict: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/strokepredict', methods=['POST'])
def stroke_predict():
    try:
        # Parse JSON input
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid input data'}), 400

        # Call prediction function
        stroke_result = predict_disease_stroke(data)

        # Check for errors in the result
        if 'error' in stroke_result:
            return jsonify(stroke_result), 400

        return jsonify(stroke_result)
    except Exception as e:
        app.logger.error(f"Error in /strokepredict: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    app.run(debug=True)
