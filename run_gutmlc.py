#!/usr/bin/env python3
"""
Quick run script for GutMLC analysis
"""

import json
import sys
from pathlib import Path

# Import your GutMLC modules
try:
    from gutmlc_python_model import GutMLCPipeline
    from data_analysis_utils import run_comprehensive_analysis
except ImportError as e:
    print(f"Error importing GutMLC modules: {e}")
    print("Make sure all Python files are in the same directory")
    sys.exit(1)

def load_config():
    """Load configuration"""
    config_file = Path("gutmlc_config.json")
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        print("Configuration file not found. Run setup first.")
        return None

def quick_run():
    """Quick model training and evaluation"""
    print("=== GUTMLC QUICK RUN ===")
    
    config = load_config()
    if config is None:
        return
    
    # Initialize pipeline
    gutmlc = GutMLCPipeline()
    
    # Use files from config
    file_paths = config['data_files']
    print(f"Using {len(file_paths)} data files")
    
    try:
        # Prepare data
        X, y, sample_info = gutmlc.prepare_data(file_paths)
        
        # Train model with config parameters
        history = gutmlc.train_model(
            X, y,
            validation_split=config['model_params']['validation_split'],
            epochs=config['model_params']['epochs'],
            batch_size=config['model_params']['batch_size']
        )
        
        # Quick evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config['model_params']['test_split'],
            random_state=config['model_params']['random_state']
        )
        
        results = gutmlc.evaluate_model(X_test, y_test)
        
        # Plot results
        gutmlc.plot_training_history()
        
        print("\n=== QUICK RUN COMPLETED ===")
        print(f"Hamming Loss: {results['hamming_loss']:.4f}")
        print(f"Average Precision: {results['average_precision']:.4f}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

def comprehensive_run():
    """Comprehensive analysis"""
    print("=== GUTMLC COMPREHENSIVE ANALYSIS ===")
    
    try:
        results = run_comprehensive_analysis()
        if results:
            print("\nComprehensive analysis completed successfully!")
            return results
        else:
            print("Comprehensive analysis failed!")
            return None
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run GutMLC analysis')
    parser.add_argument('--mode', choices=['quick', 'comprehensive'], 
                       default='quick', help='Analysis mode')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        quick_run()
    else:
        comprehensive_run()
