"""
Create a simple demo model for testing the frontend
This allows immediate testing while the real model is training
"""

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

def create_demo_model():
    """Create a simple demo model for frontend testing"""
    
    print("🔧 Creating demo model for frontend testing...")
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Create a simple sequential model similar to your actual model
    model = keras.Sequential([
        keras.layers.Input(shape=(100,)),
        keras.layers.Reshape((100, 1)),
        
        # Simple CNN layers
        keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dropout(0.5),
        
        # Dense layers
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        
        # Output layer - 10 diseases for demo
        keras.layers.Dense(10, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy', 'precision', 'recall']
    )
    
    print(f"✓ Demo model architecture created")
    model.summary()
    
    # Generate some dummy training data
    print("🎯 Generating demo training data...")
    np.random.seed(42)  # For reproducible results
    
    # Create 5 disease patterns based on our sample data
    disease_patterns = {
        0: [21.13, 12.28, 11.01, 7.98, 7.81],  # COVID-19 pattern
        1: [5.36, 3.14, 2.22, 2.14, 1.77],     # Diabetes pattern  
        2: [22.17, 12.28, 6.37, 5.20, 4.12],   # Obesity pattern
        3: [67.81, 7.67, 5.00, 2.77, 1.27],    # Hypertension pattern
        4: [13.89, 9.67, 7.89, 6.23, 5.77]     # Crohn's pattern
    }
    
    # Generate training samples
    X_train = []
    y_train = []
    
    for disease_id, pattern in disease_patterns.items():
        for _ in range(50):  # 50 samples per disease
            # Create a sample based on the pattern
            sample = np.zeros(100)
            
            # Add the main pattern with noise
            for i, val in enumerate(pattern):
                if i < 100:
                    sample[i] = val + np.random.normal(0, val * 0.1)  # 10% noise
            
            # Add some random background organisms
            remaining_indices = list(range(len(pattern), 100))
            for _ in range(np.random.randint(10, 30)):  # 10-30 additional organisms
                idx = np.random.choice(remaining_indices)
                sample[idx] = max(0, np.random.exponential(0.5))  # Exponential distribution
            
            # Ensure non-negative values
            sample = np.maximum(sample, 0)
            
            # Create multi-label target (one-hot for this disease)
            target = np.zeros(10)
            target[disease_id] = 1
            
            # Add some cross-disease associations (realistic)
            if disease_id == 1 and np.random.random() < 0.3:  # Diabetes -> Obesity
                target[2] = 1
            elif disease_id == 2 and np.random.random() < 0.4:  # Obesity -> Diabetes  
                target[1] = 1
            elif disease_id == 3 and np.random.random() < 0.2:  # Hypertension -> Cardiovascular
                target[5] = 1
            
            X_train.append(sample)
            y_train.append(target)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"✓ Generated {len(X_train)} training samples")
    print(f"  - Input shape: {X_train.shape}")
    print(f"  - Output shape: {y_train.shape}")
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("🚀 Training demo model...")
    
    # Train the model (quick training)
    history = model.fit(
        X_train_scaled, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    print("✓ Demo model training completed!")
    
    # Save the model
    model_path = 'models/gutmlc_model.h5'
    model.save(model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save the scaler
    scaler_path = 'models/scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to: {scaler_path}")
    
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
    
    classes_path = 'models/disease_classes.json'
    with open(classes_path, 'w') as f:
        json.dump(disease_classes, f, indent=2)
    print(f"✓ Disease classes saved to: {classes_path}")
    
    # Test prediction with our sample data
    print("\n🧪 Testing predictions with sample data...")
    
    # Test COVID-19 sample
    covid_sample = np.array([[21.13, 12.28, 11.01, 7.98, 7.81, 7.76, 5.47, 3.47, 2.31, 1.85] + [0.0] * 90])
    covid_scaled = scaler.transform(covid_sample)
    covid_pred = model.predict(covid_scaled, verbose=0)
    
    print("COVID-19 Sample Predictions:")
    for i, (disease, prob) in enumerate(zip(disease_classes, covid_pred[0])):
        if prob > 0.1:  # Only show significant predictions
            print(f"  • {disease}: {prob*100:.1f}%")
    
    print(f"\n🎉 Demo model setup complete!")
    print(f"📁 Files created:")
    print(f"   - {model_path}")
    print(f"   - {scaler_path}") 
    print(f"   - {classes_path}")
    print(f"\n🌐 Your frontend should now work at: http://127.0.0.1:5000")
    print(f"💡 Restart the Flask app to load the new model")

if __name__ == '__main__':
    create_demo_model()