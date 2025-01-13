from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all necessary functions
from utils.Asthma_model import predict_asthma
from utils.heart_model import predict_disease_heart
from utils.diabetes_util import predict_disease_diabetes
from utils.stroke_util import predict_disease_stroke
from main import app

# Keep your existing CORS configuration from main.py
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",
            "http://localhost:3000",
            "https://disease-prediction-app.vercel.app/",
            "https://disease-prediction-app.vercel.app",
            "*",
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Range", "X-Content-Range"],
        "supports_credentials": True,
        "max_age": 600
    }
})

# Error handling for Vercel environment
@app.errorhandler(500)
def server_error(e):
    return jsonify(error="Internal server error"), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify(error="Route not found"), 404

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(error=str(e)), 500

# Health check endpoint for Vercel
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify(status="healthy"), 200

# Required for Vercel
app = app