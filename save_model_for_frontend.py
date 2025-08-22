"""
Save trained model components for the Flask frontend
This script should be run after training to prepare the model for the web interface
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from gutmlc_python_model import GutMLCPipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have trained the model first")
    sys.exit(1)

def save_model_components():
    """Save model, scaler, and disease classes for frontend use"""
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    try:
        # Try to load the trained classifier
        classifier = GutMLCPipeline()
        
        # Check if model exists from training
        if hasattr(classifier, 'model') and classifier.model is not None:
            # Save the Keras model
            model_path = models_dir / 'gutmlc_model.h5'
            classifier.model.model.save(str(model_path))
            print(f"✓ Model saved to {model_path}")
            
            # Save scaler if available
            if hasattr(classifier, 'scaler') and classifier.scaler is not None:
                scaler_path = models_dir / 'scaler.pkl'
                with open(scaler_path, 'wb') as f:
                    pickle.dump(classifier.scaler, f)
                print(f"✓ Scaler saved to {scaler_path}")
            else:
                # Create a dummy scaler for the frontend
                scaler = StandardScaler()
                # Fit with dummy data
                dummy_data = np.random.rand(100, 100)
                scaler.fit(dummy_data)
                
                scaler_path = models_dir / 'scaler.pkl'
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                print(f"✓ Dummy scaler saved to {scaler_path}")
            
            # Save disease classes
            disease_classes = [
                'Adenoma', 'Alzheimer Disease', 'Anemia, Sickle Cell', 'Anorexia',
                'Arthritis, Juvenile', 'Arthritis, Reactive', 'Arthritis, Rheumatoid',
                'Asthma', 'Atherosclerosis', 'Attention Deficit Disorder with Hyperactivity',
                'Autism Spectrum Disorder', 'Autistic Disorder', 'Autoimmune Diseases',
                'Behcet Syndrome', 'Bipolar Disorder', 'Blastocystis Infections',
                'Breast Neoplasms', 'COVID-19', 'Cardiovascular Diseases', 'Celiac Disease',
                'Cholangitis, Sclerosing', 'Cholecystectomy', 'Cholelithiasis',
                'Chronic Periodontitis', 'Clostridium Infections', 'Clostridium difficile',
                'Cognitive Dysfunction', 'Colitis, Ulcerative', 'Colonic Diseases',
                'Colorectal Neoplasms', 'Constipation', 'Crohn Disease', 'Cystic Fibrosis',
                'Depression', 'Diabetes Mellitus', 'Diabetes Mellitus, Type 1',
                'Diabetes Mellitus, Type 2', 'Diabetes, Gestational', 'Diarrhea',
                'Enterocolitis, Necrotizing', 'Epilepsy', 'Fatigue Syndrome, Chronic',
                'Gastroesophageal Reflux', 'Gingivitis', 'Graves Disease', 'HIV',
                'HIV Infections', 'HIV-1', 'Hashimoto Disease', 'Hepatitis B virus',
                'Hypertension', 'Immunoglobulin G4-Related Disease',
                'Infant, Low Birth Weight', 'Infant, Premature',
                'Inflammatory Bowel Diseases', 'Intestinal Diseases',
                'Irritable Bowel Syndrome', 'Kidney Diseases', 'Kidney Failure, Chronic',
                'Liver Diseases', 'Lung Diseases', 'Melanoma', 'Metabolic Syndrome',
                'Migraine Disorders', 'Non-alcoholic Fatty Liver Disease', 'Obesity',
                'Obesity, Morbid', 'Overweight', 'Phenylketonurias', 'Pneumonia',
                'Precursor Cell Lymphoblastic Leukemia-Lymphoma', 'Pregnant Women',
                'Prehypertension', 'Psoriasis', 'REM Sleep Behavior Disorder',
                'Rett Syndrome', 'Rotavirus Infections', 'Schizophrenia',
                'Scleroderma, Systemic', 'Severe Acute Malnutrition',
                'Shiga-Toxigenic Escherichia coli', 'Short Bowel Syndrome',
                'Spondylarthritis', 'Spondylitis, Ankylosing', 'Thinness',
                'Thyroid Diseases', 'Thyroid Neoplasms', 'Tuberculosis',
                'Uveomeningoencephalitic Syndrome'
            ]
            
            classes_path = models_dir / 'disease_classes.json'
            with open(classes_path, 'w') as f:
                json.dump(disease_classes, f, indent=2)
            print(f"✓ Disease classes saved to {classes_path}")
            
            print("\n🎉 All model components saved successfully!")
            print("You can now run the Flask app with: python app.py")
            
        else:
            print("❌ No trained model found. Please train the model first using:")
            print("   python run_gutmlc.py --mode quick")
            
    except Exception as e:
        print(f"❌ Error saving model components: {e}")
        print("\nCreating minimal files for testing...")
        
        # Create minimal files for testing
        create_minimal_model_files(models_dir)

def create_minimal_model_files(models_dir):
    """Create minimal model files for testing the frontend"""
    
    # Create a dummy scaler
    scaler = StandardScaler()
    dummy_data = np.random.rand(100, 100)
    scaler.fit(dummy_data)
    
    scaler_path = models_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Test scaler created at {scaler_path}")
    
    # Create disease classes
    disease_classes = [
        'COVID-19', 'Anorexia', 'Hypertension', 'Melanoma', 'Tuberculosis',
        'Diabetes Mellitus', 'Obesity', 'Depression', 'Asthma', 'Arthritis, Rheumatoid'
    ]
    
    classes_path = models_dir / 'disease_classes.json'
    with open(classes_path, 'w') as f:
        json.dump(disease_classes, f, indent=2)
    print(f"✓ Test disease classes created at {classes_path}")
    
    print("\n⚠️  Note: No trained model available. The frontend will show a warning.")
    print("   Train a model first with: python run_gutmlc.py --mode quick")

if __name__ == '__main__':
    save_model_components()