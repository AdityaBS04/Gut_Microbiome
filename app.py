"""
Flask Web Application for Gut Microbiome Classification
Simple frontend for testing the GutMLC model
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'gutmlc-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'txt'}
MODEL_PATH = 'models'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Global variables for model and scaler
model = None
scaler = None
disease_classes = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_and_scaler():
    """Load the trained model and preprocessing components"""
    global model, scaler, disease_classes
    
    try:
        # Try to load saved model
        model_file = os.path.join(MODEL_PATH, 'gutmlc_model.h5')
        scaler_file = os.path.join(MODEL_PATH, 'scaler.pkl')
        classes_file = os.path.join(MODEL_PATH, 'disease_classes.json')
        
        if os.path.exists(model_file):
            model = tf.keras.models.load_model(model_file)
            logger.info("Model loaded successfully")
        else:
            logger.warning("No trained model found. Please train the model first.")
            
        if os.path.exists(scaler_file):
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("Scaler loaded successfully")
        else:
            logger.warning("No scaler found. Using default StandardScaler.")
            scaler = StandardScaler()
            
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                disease_classes = json.load(f)
            logger.info(f"Loaded {len(disease_classes)} disease classes")
        else:
            logger.warning("No disease classes found. Using default classes.")
            disease_classes = [
                'Adenoma', 'Alzheimer Disease', 'Anemia, Sickle Cell', 'Anorexia',
                'Arthritis, Juvenile', 'Arthritis, Reactive', 'Arthritis, Rheumatoid',
                'Asthma', 'Atherosclerosis', 'Attention Deficit Disorder with Hyperactivity',
                'Autism Spectrum Disorder', 'Autistic Disorder', 'Autoimmune Diseases',
                'Behcet Syndrome', 'Bipolar Disorder', 'COVID-19', 'Diabetes Mellitus'
            ]
            
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")

def preprocess_data(data):
    """Preprocess input data for prediction"""
    try:
        # Ensure we have the right number of features
        if data.shape[1] != 100:  # Assuming 100 features after preprocessing
            logger.warning(f"Input has {data.shape[1]} features, expected 100")
            # Pad or truncate to 100 features
            if data.shape[1] < 100:
                data = np.pad(data, ((0, 0), (0, 100 - data.shape[1])), mode='constant')
            else:
                data = data[:, :100]
        
        # Scale the data
        if scaler:
            data = scaler.transform(data)
        else:
            # Basic standardization if no scaler
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
            
        return data
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return data

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', disease_classes=disease_classes)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 400
        
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                
                # Read CSV file
                try:
                    df = pd.read_csv(filepath)
                    
                    # Check if CSV has feature,abundance format (rows as features)
                    if df.shape[1] == 2 and 'abundance' in df.columns:
                        # Transpose: features are rows, need them as columns
                        X = df['abundance'].values.reshape(1, -1)  # Single sample with features as columns
                        logger.info(f"Parsed CSV with {len(df)} features as a single sample")
                    
                    # Standard format: rows are samples, columns are features
                    elif df.shape[1] > 1:
                        X = df.iloc[:, 1:].values  # Skip first column (sample ID)
                        logger.info(f"Parsed CSV with {df.shape[0]} samples, {df.shape[1]-1} features")
                    else:
                        X = df.values
                        logger.info(f"Parsed CSV with shape {df.shape}")
                        
                except Exception as e:
                    return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
                    
                # Clean up uploaded file
                os.remove(filepath)
                
        # Handle manual input
        elif 'manual_input' in request.form:
            try:
                manual_data = request.form['manual_input'].strip()
                if not manual_data:
                    return jsonify({'error': 'Manual input is empty'}), 400
                    
                # Parse comma-separated values
                values = []
                for x in manual_data.split(','):
                    try:
                        val = float(x.strip())
                        values.append(val)
                    except ValueError:
                        return jsonify({'error': f'Invalid number: "{x.strip()}"'}), 400
                
                if len(values) == 0:
                    return jsonify({'error': 'No valid numbers found in input'}), 400
                    
                X = np.array([values])
                logger.info(f"Parsed manual input with {len(values)} features")
                
            except Exception as e:
                return jsonify({'error': f'Error parsing manual input: {str(e)}'}), 400
                
        else:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Preprocess data
        X_processed = preprocess_data(X)
        
        # Make prediction
        predictions = model.predict(X_processed)
        
        # Process predictions
        results = []
        for i, pred in enumerate(predictions):
            # Convert to percentages
            pred_percentages = pred * 100
            
            # Find the top disease (highest probability)
            top_disease_idx = np.argmax(pred_percentages)
            
            # Apply modifications: increase top by 20%, reduce others by 10%
            modified_pred = pred_percentages.copy()
            for j in range(len(modified_pred)):
                if j == top_disease_idx:
                    # Increase top disease by 20%
                    modified_pred[j] = min(modified_pred[j] * 1.2, 100.0)
                else:
                    # Reduce others by 10%
                    modified_pred[j] = modified_pred[j] * 0.9
                
                # Set values under 10% to 0
                if modified_pred[j] < 10.0:
                    modified_pred[j] = 0.0
            
            # Get top 5 predictions after modification
            top_indices = np.argsort(modified_pred)[-5:][::-1]
            top_diseases = []
            
            for idx in top_indices:
                if idx < len(disease_classes) and modified_pred[idx] > 0:
                    disease = disease_classes[idx]
                    probability = float(modified_pred[idx])
                    top_diseases.append({
                        'disease': disease,
                        'probability': round(probability, 2)
                    })
            
            # If no diseases are above threshold, show top disease anyway
            if not top_diseases and top_disease_idx < len(disease_classes):
                disease = disease_classes[top_disease_idx]
                probability = float(modified_pred[top_disease_idx])
                top_diseases.append({
                    'disease': disease,
                    'probability': round(max(probability, 10.0), 2)  # Ensure at least 10%
                })
            
            results.append({
                'sample_id': i + 1,
                'predictions': top_diseases
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/sample_data')
def sample_data():
    """Provide sample data for testing"""
    # Generate some sample microbiome data
    sample = np.random.rand(100) * 10  # 100 random features
    return jsonify({
        'sample_data': sample.tolist(),
        'description': 'Sample gut microbiome abundance data (100 features)'
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'disease_classes': len(disease_classes)
    }
    return jsonify(status)

if __name__ == '__main__':
    # Load model and components at startup
    load_model_and_scaler()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)