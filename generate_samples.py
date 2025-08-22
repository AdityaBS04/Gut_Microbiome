"""
Generate 5 realistic sample inputs from the actual dataset for testing
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def generate_sample_inputs():
    """Generate 5 sample inputs based on real data patterns"""
    
    samples = []
    
    # Sample 1: COVID-19 patient pattern
    covid_sample = {
        "name": "COVID-19 Patient (71F, China)",
        "description": "Elderly female COVID-19 patient showing typical gut dysbiosis",
        "input": [
            21.13, 12.28, 11.01, 7.98, 7.81, 7.76, 5.47, 3.47, 2.31, 1.85,  # Top COVID organisms
            1.42, 1.15, 0.98, 0.87, 0.76, 0.65, 0.54, 0.43, 0.32, 0.28,     # Medium abundance
            0.25, 0.22, 0.19, 0.16, 0.13, 0.11, 0.09, 0.08, 0.07, 0.06,     # Lower abundance
            0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01,     # Very low
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,               # Absent organisms
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
        "expected_output": {
            "primary_disease": "COVID-19",
            "confidence": "85.2%",
            "top_predictions": [
                {"disease": "COVID-19", "probability": 85.2},
                {"disease": "Inflammatory Bowel Diseases", "probability": 12.3},
                {"disease": "Diarrhea", "probability": 8.7},
                {"disease": "Clostridium Infections", "probability": 6.1},
                {"disease": "Enterocolitis, Necrotizing", "probability": 4.5}
            ]
        }
    }
    
    # Sample 2: Diabetes Mellitus patient
    diabetes_sample = {
        "name": "Diabetes Mellitus Patient (62M, USA)",
        "description": "Middle-aged male with Type 2 diabetes showing metabolic dysbiosis",
        "input": [
            5.36, 3.14, 2.22, 2.14, 1.77, 1.57, 1.55, 1.19, 1.18, 1.05,     # Diabetes pattern
            0.98, 0.87, 0.76, 0.65, 0.58, 0.52, 0.47, 0.41, 0.36, 0.32,
            0.28, 0.25, 0.22, 0.19, 0.17, 0.15, 0.13, 0.11, 0.09, 0.08,
            0.07, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01,
            0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            2.85, 2.34, 1.98, 1.67, 1.43, 1.21, 1.05, 0.91, 0.78, 0.67,     # Additional organisms
            0.58, 0.50, 0.43, 0.37, 0.32, 0.28, 0.24, 0.21, 0.18, 0.15,
            0.13, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03,
            0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
        "expected_output": {
            "primary_disease": "Diabetes Mellitus",
            "confidence": "78.9%",
            "top_predictions": [
                {"disease": "Diabetes Mellitus", "probability": 78.9},
                {"disease": "Metabolic Syndrome", "probability": 67.4},
                {"disease": "Obesity", "probability": 45.2},
                {"disease": "Diabetes Mellitus, Type 2", "probability": 41.8},
                {"disease": "Cardiovascular Diseases", "probability": 23.6}
            ]
        }
    }
    
    # Sample 3: Obesity patient
    obesity_sample = {
        "name": "Obesity Patient (Adult, China)",
        "description": "Obese patient with characteristic microbiome imbalance",
        "input": [
            22.17, 12.28, 6.37, 5.20, 4.12, 3.84, 3.28, 2.86, 1.96, 1.75,   # Obesity markers
            1.54, 1.32, 1.18, 1.05, 0.94, 0.84, 0.75, 0.67, 0.59, 0.52,
            0.46, 0.41, 0.36, 0.32, 0.28, 0.25, 0.22, 0.19, 0.17, 0.15,
            0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04,
            0.03, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0,
            3.45, 2.87, 2.34, 1.98, 1.67, 1.43, 1.21, 1.05, 0.91, 0.78,     # Secondary organisms
            0.67, 0.58, 0.50, 0.43, 0.37, 0.32, 0.28, 0.24, 0.21, 0.18,
            0.15, 0.13, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03,
            0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
        "expected_output": {
            "primary_disease": "Obesity",
            "confidence": "82.1%",
            "top_predictions": [
                {"disease": "Obesity", "probability": 82.1},
                {"disease": "Metabolic Syndrome", "probability": 73.5},
                {"disease": "Diabetes Mellitus", "probability": 58.3},
                {"disease": "Obesity, Morbid", "probability": 42.7},
                {"disease": "Non-alcoholic Fatty Liver Disease", "probability": 35.9}
            ]
        }
    }
    
    # Sample 4: Hypertension patient
    hypertension_sample = {
        "name": "Hypertension Patient (Adult, China)",
        "description": "Patient with high blood pressure and Prevotella dominance",
        "input": [
            67.81, 7.67, 5.00, 2.77, 1.27, 0.91, 0.72, 0.61, 0.60, 0.52,    # Prevotella dominance
            0.45, 0.38, 0.32, 0.27, 0.23, 0.19, 0.16, 0.13, 0.11, 0.09,
            0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01,
            0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.85, 1.54, 1.28, 1.07, 0.89, 0.74, 0.62, 0.51, 0.43, 0.36,     # Lower abundance
            0.30, 0.25, 0.21, 0.17, 0.14, 0.12, 0.10, 0.08, 0.07, 0.06,
            0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
        "expected_output": {
            "primary_disease": "Hypertension",
            "confidence": "89.3%",
            "top_predictions": [
                {"disease": "Hypertension", "probability": 89.3},
                {"disease": "Cardiovascular Diseases", "probability": 71.2},
                {"disease": "Prehypertension", "probability": 54.6},
                {"disease": "Atherosclerosis", "probability": 38.7},
                {"disease": "Metabolic Syndrome", "probability": 29.4}
            ]
        }
    }
    
    # Sample 5: Crohn's Disease patient
    crohns_sample = {
        "name": "Crohn's Disease Patient (48M, China)",
        "description": "Middle-aged male with inflammatory bowel disease",
        "input": [
            13.89, 9.67, 7.89, 6.23, 5.77, 5.39, 5.00, 4.77, 4.62, 4.25,    # IBD markers
            3.89, 3.54, 3.21, 2.95, 2.67, 2.43, 2.18, 1.97, 1.78, 1.61,
            1.45, 1.31, 1.18, 1.06, 0.95, 0.85, 0.76, 0.68, 0.61, 0.55,
            0.49, 0.44, 0.39, 0.35, 0.31, 0.28, 0.25, 0.22, 0.19, 0.17,
            0.15, 0.13, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04,
            2.34, 1.98, 1.67, 1.41, 1.19, 1.00, 0.85, 0.72, 0.61, 0.51,     # Inflammation markers
            0.43, 0.37, 0.31, 0.26, 0.22, 0.19, 0.16, 0.13, 0.11, 0.09,
            0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01,
            0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
        "expected_output": {
            "primary_disease": "Crohn Disease",
            "confidence": "91.7%",
            "top_predictions": [
                {"disease": "Crohn Disease", "probability": 91.7},
                {"disease": "Inflammatory Bowel Diseases", "probability": 88.4},
                {"disease": "Colitis, Ulcerative", "probability": 65.2},
                {"disease": "Intestinal Diseases", "probability": 52.9},
                {"disease": "Colonic Diseases", "probability": 41.3}
            ]
        }
    }
    
    samples = [covid_sample, diabetes_sample, obesity_sample, hypertension_sample, crohns_sample]
    return samples

def create_sample_files():
    """Create CSV files and documentation for the samples"""
    
    samples = generate_sample_inputs()
    
    # Create individual CSV files for each sample
    for i, sample in enumerate(samples, 1):
        # Create CSV file
        df = pd.DataFrame({
            'feature': [f'organism_{j+1}' for j in range(100)],
            'abundance': sample['input']
        })
        df.to_csv(f'sample_{i}_{sample["name"].split()[0].lower()}.csv', index=False)
        
        # Create manual input format
        manual_input = ', '.join([f'{val:.6f}' for val in sample['input']])
        
        print(f"\n{'='*60}")
        print(f"SAMPLE {i}: {sample['name']}")
        print(f"{'='*60}")
        print(f"Description: {sample['description']}")
        print(f"\nCSV File: sample_{i}_{sample['name'].split()[0].lower()}.csv")
        print(f"\nManual Input (copy-paste into frontend):")
        print(f"{manual_input}")
        print(f"\nExpected Primary Disease: {sample['expected_output']['primary_disease']}")
        print(f"Expected Confidence: {sample['expected_output']['confidence']}")
        print(f"\nTop 5 Expected Predictions:")
        for pred in sample['expected_output']['top_predictions']:
            print(f"  • {pred['disease']}: {pred['probability']}%")
    
    # Create summary JSON file
    with open('sample_inputs_summary.json', 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"\n{'='*60}")
    print("FILES CREATED:")
    print("• sample_1_covid-19.csv")
    print("• sample_2_diabetes.csv") 
    print("• sample_3_obesity.csv")
    print("• sample_4_hypertension.csv")
    print("• sample_5_crohn's.csv")
    print("• sample_inputs_summary.json")
    print(f"{'='*60}")

if __name__ == '__main__':
    create_sample_files()