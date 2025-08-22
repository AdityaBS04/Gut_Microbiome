"""
Create a simple demo model for testing the frontend - simplified version
"""

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

def create_simple_demo():
    print("Creating demo model for frontend testing...")
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Create a very simple model
    model = keras.Sequential([
        keras.layers.Input(shape=(100,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='sigmoid')  # 10 diseases
    ])
    
    # Compile with simpler metrics
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Demo model created")
    print(f"Model has {model.count_params()} parameters")
    
    # Generate simple training data
    np.random.seed(42)
    
    # Create 100 samples with realistic patterns
    X_train = []
    y_train = []
    
    # Disease patterns based on our samples
    patterns = {
        0: [21.13, 12.28, 11.01, 7.98, 7.81],  # COVID
        1: [5.36, 3.14, 2.22, 2.14, 1.77],     # Diabetes  
        2: [22.17, 12.28, 6.37, 5.20, 4.12],   # Obesity
        3: [67.81, 7.67, 5.00, 2.77, 1.27],    # Hypertension
        4: [13.89, 9.67, 7.89, 6.23, 5.77]     # Crohn's
    }
    
    for disease_id, pattern in patterns.items():
        for _ in range(20):  # 20 samples per disease
            sample = np.zeros(100)
            
            # Add pattern with some noise
            for i, val in enumerate(pattern):
                if i < 100:
                    sample[i] = max(0, val + np.random.normal(0, val * 0.2))
            
            # Add random background
            for i in range(len(pattern), 100):
                if np.random.random() < 0.3:  # 30% chance of having value
                    sample[i] = max(0, np.random.exponential(1))
            
            # Create target
            target = np.zeros(10)
            target[disease_id] = 1
            
            X_train.append(sample)
            y_train.append(target)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"Generated {len(X_train)} training samples")
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("Training model...")
    
    # Quick training
    model.fit(X_train_scaled, y_train, epochs=5, batch_size=16, verbose=1)
    
    print("Training completed!")
    
    # Save model
    model.save('models/gutmlc_model.h5')
    print("Model saved")
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved")
    
    # Save disease classes
    disease_classes = [
        'COVID-19',
        'Diabetes Mellitus', 
        'Obesity',
        'Hypertension',
        'Crohn Disease',
        'Cardiovascular Diseases',
        'Depression',
        'Asthma',
        'Arthritis, Rheumatoid',
        'Inflammatory Bowel Diseases'
    ]
    
    with open('models/disease_classes.json', 'w') as f:
        json.dump(disease_classes, f, indent=2)
    print("Disease classes saved")
    
    # Test with COVID sample
    covid_sample = np.array([[21.13, 12.28, 11.01, 7.98, 7.81] + [0.0] * 95])
    covid_scaled = scaler.transform(covid_sample)
    pred = model.predict(covid_scaled, verbose=0)
    
    print("Test prediction:")
    for i, (disease, prob) in enumerate(zip(disease_classes, pred[0])):
        if prob > 0.1:
            print(f"  {disease}: {prob*100:.1f}%")
    
    print("Demo model setup complete!")
    print("Restart Flask app to use the new model")

if __name__ == '__main__':
    create_simple_demo()