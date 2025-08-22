"""
Setup and Configuration Script for GutMLC
Easy setup and configuration for running the gut microbiota classification model
"""

import os
import sys
import glob
import subprocess
import pkg_resources
from pathlib import Path

# Required packages with versions
REQUIRED_PACKAGES = {
    'numpy': '>=1.21.0',
    'pandas': '>=1.3.0', 
    'scikit-learn': '>=1.0.0',
    'tensorflow': '>=2.8.0',
    'matplotlib': '>=3.5.0',
    'seaborn': '>=0.11.0',
    'scipy': '>=1.7.0',
    'networkx': '>=2.6.0',
    'imbalanced-learn': '>=0.8.0',
    'statsmodels': '>=0.13.0'
}

class GutMLCSetup:
    """Setup and configuration class for GutMLC"""
    
    def __init__(self):
        self.project_dir = Path.cwd()
        self.data_dir = self.project_dir / "data"
        self.models_dir = self.project_dir / "models"
        self.results_dir = self.project_dir / "results"
        self.config = {}
        
    def check_dependencies(self):
        """Check if all required packages are installed"""
        print("=== CHECKING DEPENDENCIES ===")
        
        missing_packages = []
        
        for package, version in REQUIRED_PACKAGES.items():
            try:
                pkg_resources.require(f"{package}{version}")
                print(f"✓ {package} {version}")
            except pkg_resources.DistributionNotFound:
                missing_packages.append(package)
                print(f"✗ {package} {version} - NOT INSTALLED")
            except pkg_resources.VersionConflict as e:
                print(f"⚠ {package} {version} - VERSION CONFLICT: {e}")
                
        if missing_packages:
            print(f"\\nMissing packages: {missing_packages}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        print("\\n✓ All dependencies satisfied!")
        return True
    
    def setup_directories(self):
        """Create necessary project directories"""
        print("=== SETTING UP DIRECTORIES ===")
        
        directories = [self.data_dir, self.models_dir, self.results_dir]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            print(f"✓ Created/verified: {directory}")
    
    def discover_data_files(self):
        """Automatically discover CSV data files"""
        print("=== DISCOVERING DATA FILES ===")
        
        # Look for CSV files in current directory and data directory
        csv_files = []
        
        # Current directory
        current_csvs = glob.glob("*.csv")
        csv_files.extend(current_csvs)
        
        # Data directory
        if self.data_dir.exists():
            data_csvs = glob.glob(str(self.data_dir / "*.csv"))
            csv_files.extend(data_csvs)
        
        print(f"Found {len(csv_files)} CSV files:")
        for i, file in enumerate(csv_files, 1):
            print(f"  {i}. {file}")
            
        return csv_files
    
    def analyze_data_files(self, csv_files):
        """Analyze discovered CSV files to understand structure"""
        print("=== ANALYZING DATA FILES ===")
        
        import pandas as pd
        
        file_info = {}
        
        for file in csv_files:
            try:
                # Read first few rows to understand structure
                df = pd.read_csv(file, nrows=5)
                
                file_info[file] = {
                    'columns': list(df.columns),
                    'shape_sample': len(df),
                    'has_disease_name': 'disease_name' in df.columns,
                    'has_scientific_name': 'scientific_name' in df.columns,
                    'has_relative_abundance': 'relative_abundance' in df.columns,
                    'has_run_id': 'run_id' in df.columns
                }
                
                if file_info[file]['has_disease_name']:
                    disease = df['disease_name'].iloc[0] if len(df) > 0 else 'Unknown'
                    file_info[file]['disease'] = disease
                    print(f"✓ {file}: {disease}")
                else:
                    print(f"⚠ {file}: No disease_name column found")
                    
            except Exception as e:
                print(f"✗ {file}: Error reading file - {e}")
                file_info[file] = {'error': str(e)}
        
        return file_info
    
    def create_config_file(self, csv_files, file_info):
        """Create configuration file for the project"""
        print("=== CREATING CONFIGURATION FILE ===")
        
        # Filter valid files
        valid_files = [f for f in csv_files if f in file_info and 'error' not in file_info[f]]
        
        config = {
            'project_name': 'GutMLC_Disease_Classification',
            'data_files': valid_files,
            'model_params': {
                'batch_size': 32,
                'epochs': 100,
                'learning_rate': 0.001,
                'validation_split': 0.2,
                'test_split': 0.2,
                'random_state': 42
            },
            'preprocessing': {
                'apply_log_transform': True,
                'apply_standardization': True,
                'svd_components': 100,
                'variance_threshold': 0.01,
                'correlation_threshold': 0.9
            },
            'augmentation': {
                'apply_smote': True,
                'smote_strategy': 'auto',
                'add_noise': False,
                'noise_factor': 0.01
            },
            'feature_selection': {
                'remove_low_variance': True,
                'remove_high_correlation': True,
                'differential_abundance': True
            },
            'model_architecture': {
                'conv_layers': [64, 64, 128, 128, 256, 256],
                'dense_layers': [512, 256],
                'dropout_rates': [0.25, 0.25, 0.5, 0.5, 0.3],
                'activation': 'relu',
                'final_activation': 'sigmoid'
            },
            'directories': {
                'data': str(self.data_dir),
                'models': str(self.models_dir),
                'results': str(self.results_dir)
            }
        }
        
        # Save config
        import json
        config_file = self.project_dir / "gutmlc_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Configuration saved to: {config_file}")
        self.config = config
        return config
    
    def create_run_script(self):
        """Create a simple run script"""
        print("=== CREATING RUN SCRIPT ===")
        
        run_script_content = '''#!/usr/bin/env python3
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
        
        print("\\n=== QUICK RUN COMPLETED ===")
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
            print("\\nComprehensive analysis completed successfully!")
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
'''
        
        run_script_file = self.project_dir / "run_gutmlc.py"
        with open(run_script_file, 'w') as f:
            f.write(run_script_content)
        
        # Make executable
        os.chmod(run_script_file, 0o755)
        
        print(f"✓ Run script created: {run_script_file}")
        print("Usage:")
        print("  python run_gutmlc.py --mode quick")
        print("  python run_gutmlc.py --mode comprehensive")
    
    def full_setup(self):
        """Run complete setup process"""
        print("=== GUTMLC FULL SETUP ===\\n")
        
        # Check dependencies
        if not self.check_dependencies():
            print("\\nPlease install missing dependencies before continuing.")
            return False
        
        # Setup directories
        self.setup_directories()
        
        # Discover and analyze data
        csv_files = self.discover_data_files()
        if not csv_files:
            print("\n⚠ No CSV files found!")
            print("Please ensure your disease CSV files are in the current directory or data/ folder")
            return False
        
        file_info = self.analyze_data_files(csv_files)
        
        # Create configuration
        config = self.create_config_file(csv_files, file_info)
        
        # Create run script
        self.create_run_script()
        
        # Create requirements file
        self.create_requirements_file()
        
        # Print summary
        self.print_setup_summary(config, file_info)
        
        return True
    
    def create_requirements_file(self):
        """Create requirements.txt file"""
        print("=== CREATING REQUIREMENTS FILE ===")
        
        requirements_content = """# GutMLC Requirements
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
networkx>=2.6.0
imbalanced-learn>=0.8.0
statsmodels>=0.13.0
jupyter>=1.0.0
plotly>=5.0.0
"""
        
        requirements_file = self.project_dir / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write(requirements_content)
        
        print(f"✓ Requirements file created: {requirements_file}")
    
    def print_setup_summary(self, config, file_info):
        """Print setup summary"""
        print("\n" + "="*60)
        print("SETUP SUMMARY")
        print("="*60)
        
        # Valid files
        valid_files = [f for f in config['data_files']]
        diseases = []
        for file in valid_files:
            if file in file_info and 'disease' in file_info[file]:
                diseases.append(file_info[file]['disease'])
        
        print(f"✓ Project directory: {self.project_dir}")
        print(f"✓ Data files found: {len(valid_files)}")
        print(f"✓ Diseases identified: {len(set(diseases))}")
        print(f"✓ Unique diseases: {', '.join(set(diseases))}")
        
        print(f"\nNext steps:")
        print(f"1. Install dependencies: pip install -r requirements.txt")
        print(f"2. Quick run: python run_gutmlc.py --mode quick")
        print(f"3. Full analysis: python run_gutmlc.py --mode comprehensive")
        
        print(f"\nFiles created:")
        print(f"- gutmlc_config.json (configuration)")
        print(f"- run_gutmlc.py (execution script)")
        print(f"- requirements.txt (dependencies)")
        
        print("\n" + "="*60)

def install_dependencies():
    """Install required dependencies"""
    print("=== INSTALLING DEPENDENCIES ===")
    
    packages = list(REQUIRED_PACKAGES.keys())
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False
    
    print("✓ All dependencies installed!")
    return True

def create_example_notebook():
    """Create an example Jupyter notebook"""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# GutMLC: Gut Microbiota Multi-Label Disease Classification\n",
                    "\n",
                    "This notebook demonstrates how to use the GutMLC pipeline for predicting multiple diseases from gut microbiota profiles.\n",
                    "\n",
                    "## Overview\n",
                    "- Load and preprocess gut microbiota data\n",
                    "- Train multi-label classification model\n",
                    "- Evaluate model performance\n",
                    "- Analyze results"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Import required libraries\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import json\n",
                    "\n",
                    "# Import GutMLC modules\n",
                    "from gutmlc_python_model import GutMLCPipeline\n",
                    "from data_analysis_utils import DataAnalyzer, AdvancedFeatureSelector\n",
                    "\n",
                    "# Set style\n",
                    "plt.style.use('seaborn-v0_8')\n",
                    "sns.set_palette('husl')\n",
                    "\n",
                    "print(\"Libraries imported successfully!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load configuration\n",
                    "with open('gutmlc_config.json', 'r') as f:\n",
                    "    config = json.load(f)\n",
                    "\n",
                    "print(\"Configuration loaded:\")\n",
                    "print(f\"- Data files: {len(config['data_files'])}\")\n",
                    "print(f\"- Model epochs: {config['model_params']['epochs']}\")\n",
                    "print(f\"- Batch size: {config['model_params']['batch_size']}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Initialize pipeline and load data\n",
                    "gutmlc = GutMLCPipeline()\n",
                    "\n",
                    "# Prepare data\n",
                    "X, y, sample_info = gutmlc.prepare_data(config['data_files'])\n",
                    "\n",
                    "print(f\"Dataset prepared:\")\n",
                    "print(f\"- Samples: {X.shape[0]}\")\n",
                    "print(f\"- Features: {X.shape[1]}\")\n",
                    "print(f\"- Diseases: {y.shape[1]}\")\n",
                    "print(f\"- Disease classes: {gutmlc.preprocessor.mlb.classes_}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Data exploration\n",
                    "analyzer = DataAnalyzer()\n",
                    "\n",
                    "# Analyze feature patterns\n",
                    "feature_df = pd.DataFrame(X)\n",
                    "patterns = analyzer.analyze_microbial_patterns(feature_df, sample_info)\n",
                    "\n",
                    "# Visualize class distribution\n",
                    "class_counts = y.sum(axis=0)\n",
                    "diseases = gutmlc.preprocessor.mlb.classes_\n",
                    "\n",
                    "plt.figure(figsize=(12, 6))\n",
                    "plt.bar(range(len(diseases)), class_counts)\n",
                    "plt.xlabel('Diseases')\n",
                    "plt.ylabel('Number of Positive Samples')\n",
                    "plt.title('Disease Distribution in Dataset')\n",
                    "plt.xticks(range(len(diseases)), diseases, rotation=45, ha='right')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Train the model\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "\n",
                    "# Split data\n",
                    "X_train, X_test, y_train, y_test = train_test_split(\n",
                    "    X, y, test_size=0.2, random_state=42\n",
                    ")\n",
                    "\n",
                    "print(f\"Training set: {X_train.shape}\")\n",
                    "print(f\"Test set: {X_test.shape}\")\n",
                    "\n",
                    "# Train model\n",
                    "history = gutmlc.train_model(\n",
                    "    X_train, y_train,\n",
                    "    validation_split=0.2,\n",
                    "    epochs=50,  # Reduced for notebook\n",
                    "    batch_size=32\n",
                    ")\n",
                    "\n",
                    "print(\"Model training completed!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Evaluate model\n",
                    "results = gutmlc.evaluate_model(X_test, y_test)\n",
                    "\n",
                    "print(\"Model Performance:\")\n",
                    "print(f\"- Hamming Loss: {results['hamming_loss']:.4f}\")\n",
                    "print(f\"- Coverage Error: {results['coverage_error']:.4f}\")\n",
                    "print(f\"- Average Precision: {results['average_precision']:.4f}\")\n",
                    "\n",
                    "# Plot training history\n",
                    "gutmlc.plot_training_history()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Analyze predictions\n",
                    "y_pred = results['predictions']\n",
                    "y_pred_binary = results['binary_predictions']\n",
                    "\n",
                    "# Create confusion matrix for each disease\n",
                    "from sklearn.metrics import multilabel_confusion_matrix\n",
                    "\n",
                    "cm = multilabel_confusion_matrix(y_test, y_pred_binary)\n",
                    "\n",
                    "# Plot confusion matrices\n",
                    "n_diseases = len(diseases)\n",
                    "fig, axes = plt.subplots(2, (n_diseases + 1) // 2, figsize=(15, 8))\n",
                    "axes = axes.flatten()\n",
                    "\n",
                    "for i, disease in enumerate(diseases):\n",
                    "    if i < len(axes):\n",
                    "        sns.heatmap(cm[i], annot=True, fmt='d', ax=axes[i], cmap='Blues')\n",
                    "        axes[i].set_title(f'{disease}')\n",
                    "        axes[i].set_xlabel('Predicted')\n",
                    "        axes[i].set_ylabel('Actual')\n",
                    "\n",
                    "# Hide extra subplots\n",
                    "for i in range(len(diseases), len(axes)):\n",
                    "    axes[i].set_visible(False)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Save results\n",
                    "import pickle\n",
                    "from datetime import datetime\n",
                    "\n",
                    "timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n",
                    "\n",
                    "# Save model\n",
                    "gutmlc.model.model.save(f'models/gutmlc_model_{timestamp}.h5')\n",
                    "\n",
                    "# Save results\n",
                    "results_summary = {\n",
                    "    'timestamp': timestamp,\n",
                    "    'dataset_info': {\n",
                    "        'n_samples': X.shape[0],\n",
                    "        'n_features': X.shape[1],\n",
                    "        'n_diseases': y.shape[1],\n",
                    "        'diseases': list(diseases)\n",
                    "    },\n",
                    "    'performance': {\n",
                    "        'hamming_loss': results['hamming_loss'],\n",
                    "        'coverage_error': results['coverage_error'],\n",
                    "        'average_precision': results['average_precision']\n",
                    "    },\n",
                    "    'config': config\n",
                    "}\n",
                    "\n",
                    "with open(f'results/results_{timestamp}.json', 'w') as f:\n",
                    "    json.dump(results_summary, f, indent=2)\n",
                    "\n",
                    "print(f\"Results saved to results/results_{timestamp}.json\")\n",
                    "print(f\"Model saved to models/gutmlc_model_{timestamp}.h5\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    notebook_file = Path("gutmlc_example.ipynb")
    with open(notebook_file, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print(f"✓ Example notebook created: {notebook_file}")

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup GutMLC project')
    parser.add_argument('--install-deps', action='store_true', 
                       help='Install required dependencies')
    parser.add_argument('--create-notebook', action='store_true',
                       help='Create example Jupyter notebook')
    
    args = parser.parse_args()
    
    if args.install_deps:
        success = install_dependencies()
        if not success:
            print("Failed to install some dependencies")
            return
    
    # Run full setup
    setup = GutMLCSetup()
    success = setup.full_setup()
    
    if success:
        print("\n✅ Setup completed successfully!")
        
        if args.create_notebook:
            create_example_notebook()
            print("📓 Example notebook created!")
            
    else:
        print("\n❌ Setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()