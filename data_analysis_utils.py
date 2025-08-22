"""
Data Analysis and Model Enhancement Utilities for GutMLC
Additional tools for data exploration, feature engineering, and model optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr, spearmanr
import networkx as nx
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """Comprehensive data analysis for gut microbiota datasets"""
    
    def __init__(self):
        self.data_stats = {}
        self.correlation_matrix = None
        
    def analyze_dataset_distribution(self, df):
        """Analyze the distribution of diseases, samples, and organisms"""
        print("=== DATASET DISTRIBUTION ANALYSIS ===")
        
        # Disease distribution
        disease_counts = df['disease_name'].value_counts()
        print(f"\\nDisease distribution:")
        print(disease_counts)
        
        # Sample distribution per disease
        sample_counts = df.groupby('disease_name')['run_id'].nunique().sort_values(ascending=False)
        print(f"\\nSamples per disease:")
        print(sample_counts)
        
        # Organism diversity per disease
        organism_diversity = df.groupby('disease_name')['scientific_name'].nunique().sort_values(ascending=False)
        print(f"\\nOrganism diversity per disease:")
        print(organism_diversity)
        
        # Abundance statistics
        abundance_stats = df.groupby('disease_name')['relative_abundance'].agg(['mean', 'std', 'min', 'max'])
        print(f"\\nAbundance statistics by disease:")
        print(abundance_stats)
        
        # Store stats
        self.data_stats = {
            'disease_counts': disease_counts,
            'sample_counts': sample_counts, 
            'organism_diversity': organism_diversity,
            'abundance_stats': abundance_stats
        }
        
        return self.data_stats
    
    def plot_data_distribution(self, df):
        """Create comprehensive visualizations of data distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Disease frequency
        disease_counts = df['disease_name'].value_counts()
        axes[0, 0].bar(range(len(disease_counts)), disease_counts.values)
        axes[0, 0].set_title('Disease Frequency Distribution')
        axes[0, 0].set_xlabel('Diseases')
        axes[0, 0].set_ylabel('Number of Records')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Sample count per disease
        sample_counts = df.groupby('disease_name')['run_id'].nunique()
        axes[0, 1].bar(range(len(sample_counts)), sample_counts.values)
        axes[0, 1].set_title('Samples per Disease')
        axes[0, 1].set_xlabel('Diseases')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Abundance distribution
        axes[1, 0].hist(df['relative_abundance'], bins=50, alpha=0.7)
        axes[1, 0].set_title('Relative Abundance Distribution')
        axes[1, 0].set_xlabel('Relative Abundance')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_yscale('log')
        
        # Age distribution
        if 'host_age' in df.columns and df['host_age'].notna().sum() > 0:
            age_data = df[df['host_age'].notna()]
            sns.boxplot(data=age_data, x='disease_name', y='host_age', ax=axes[1, 1])
            axes[1, 1].set_title('Age Distribution by Disease')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No age data available', ha='center', va='center')
            axes[1, 1].set_title('Age Distribution by Disease')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_microbial_patterns(self, feature_matrix, sample_info):
        """Analyze microbial abundance patterns"""
        print("=== MICROBIAL PATTERN ANALYSIS ===")
        
        # Top abundant organisms across all samples
        mean_abundances = feature_matrix.mean(axis=0).sort_values(ascending=False)
        print(f"\\nTop 10 most abundant organisms (mean):")
        print(mean_abundances.head(10))
        
        # Most variable organisms
        std_abundances = feature_matrix.std(axis=0).sort_values(ascending=False)
        print(f"\\nTop 10 most variable organisms (std):")
        print(std_abundances.head(10))
        
        # Sparsity analysis
        sparsity = (feature_matrix == 0).sum().sum() / (feature_matrix.shape[0] * feature_matrix.shape[1])
        print(f"\\nDataset sparsity: {sparsity:.2%}")
        
        # Per-sample diversity (number of non-zero organisms)
        sample_diversity = (feature_matrix > 0).sum(axis=1)
        print(f"\\nSample diversity statistics:")
        print(f"Mean organisms per sample: {sample_diversity.mean():.1f}")
        print(f"Std organisms per sample: {sample_diversity.std():.1f}")
        print(f"Min organisms per sample: {sample_diversity.min()}")
        print(f"Max organisms per sample: {sample_diversity.max()}")
        
        return {
            'mean_abundances': mean_abundances,
            'std_abundances': std_abundances,
            'sparsity': sparsity,
            'sample_diversity': sample_diversity
        }
    
    def perform_dimensionality_analysis(self, X, y, sample_info):
        """Perform dimensionality reduction and visualization"""
        print("=== DIMENSIONALITY REDUCTION ANALYSIS ===")
        
        # PCA analysis
        pca = PCA(n_components=50)
        X_pca = pca.fit_transform(X)
        
        # Explained variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
        print(f"Components needed for 95% variance: {n_components_95}")
        
        # t-SNE for visualization (use subset if too large)
        if X.shape[0] > 1000:
            idx = np.random.choice(X.shape[0], 1000, replace=False)
            X_subset = X[idx]
            y_subset = y[idx]
        else:
            X_subset = X
            y_subset = y
            
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X_subset.shape[0]-1))
        X_tsne = tsne.fit_transform(X_subset)
        
        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # PCA explained variance
        axes[0].plot(range(1, len(cumsum_var) + 1), cumsum_var)
        axes[0].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        axes[0].axvline(x=n_components_95, color='r', linestyle='--', alpha=0.7)
        axes[0].set_xlabel('Number of Components')
        axes[0].set_ylabel('Cumulative Explained Variance')
        axes[0].set_title('PCA Explained Variance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # PCA 2D plot
        disease_labels = np.argmax(y_subset, axis=1) if y_subset.ndim > 1 else y_subset
        scatter = axes[1].scatter(X_pca[:len(X_subset), 0], X_pca[:len(X_subset), 1], 
                                 c=disease_labels, cmap='tab10', alpha=0.6)
        axes[1].set_xlabel('First Principal Component')
        axes[1].set_ylabel('Second Principal Component')
        axes[1].set_title('PCA Visualization')
        
        # t-SNE plot
        scatter = axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=disease_labels, 
                                 cmap='tab10', alpha=0.6)
        axes[2].set_xlabel('t-SNE 1')
        axes[2].set_ylabel('t-SNE 2')
        axes[2].set_title('t-SNE Visualization')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'pca_components_95': n_components_95,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'X_pca': X_pca,
            'X_tsne': X_tsne
        }

class AdvancedFeatureSelector:
    """Advanced feature selection methods for microbiome data"""
    
    def __init__(self):
        self.selected_features = None
        self.feature_scores = None
        
    def differential_abundance_analysis(self, feature_matrix, labels, disease_names):
        """Identify differentially abundant microbes for each disease"""
        print("=== DIFFERENTIAL ABUNDANCE ANALYSIS ===")
        
        significant_features = {}
        
        for i, disease in enumerate(disease_names):
            # Binary labels for current disease
            disease_labels = labels[:, i] if labels.ndim > 1 else (labels == i).astype(int)
            
            if np.sum(disease_labels) < 5:  # Skip diseases with too few samples
                continue
                
            feature_pvalues = []
            feature_effects = []
            
            for j in range(feature_matrix.shape[1]):
                feature_values = feature_matrix.iloc[:, j]
                
                # Split by disease presence
                positive_samples = feature_values[disease_labels == 1]
                negative_samples = feature_values[disease_labels == 0]
                
                if len(positive_samples) > 0 and len(negative_samples) > 0:
                    # Mann-Whitney U test for non-parametric comparison
                    from scipy.stats import mannwhitneyu
                    try:
                        statistic, pvalue = mannwhitneyu(positive_samples, negative_samples, 
                                                       alternative='two-sided')
                        effect_size = np.mean(positive_samples) - np.mean(negative_samples)
                        feature_pvalues.append(pvalue)
                        feature_effects.append(effect_size)
                    except:
                        feature_pvalues.append(1.0)
                        feature_effects.append(0.0)
                else:
                    feature_pvalues.append(1.0)
                    feature_effects.append(0.0)
            
            # Apply FDR correction
            from statsmodels.stats.multitest import fdrcorrection
            rejected, pvals_corrected = fdrcorrection(feature_pvalues, alpha=0.05)
            
            # Store significant features
            significant_idx = np.where(rejected)[0]
            if len(significant_idx) > 0:
                significant_features[disease] = {
                    'indices': significant_idx,
                    'names': feature_matrix.columns[significant_idx].tolist(),
                    'pvalues': np.array(pvals_corrected)[significant_idx],
                    'effect_sizes': np.array(feature_effects)[significant_idx]
                }
                
                print(f"\\n{disease}: {len(significant_idx)} significant features")
                # Show top 5
                top_idx = np.argsort(np.array(feature_effects)[significant_idx])[-5:]
                for idx in top_idx:
                    real_idx = significant_idx[idx]
                    print(f"  {feature_matrix.columns[real_idx]}: "
                          f"effect={feature_effects[real_idx]:.3f}, "
                          f"p={pvals_corrected[real_idx]:.3e}")
        
        return significant_features
    
    def correlation_based_selection(self, feature_matrix, threshold=0.9):
        """Remove highly correlated features"""
        print(f"=== CORRELATION-BASED FEATURE SELECTION (threshold={threshold}) ===")
        
        # Calculate correlation matrix
        corr_matrix = feature_matrix.corr().abs()
        
        # Find highly correlated pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation above threshold
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        print(f"Found {len(high_corr_features)} highly correlated features to remove")
        
        # Remove highly correlated features
        reduced_features = feature_matrix.drop(columns=high_corr_features)
        
        print(f"Features reduced from {feature_matrix.shape[1]} to {reduced_features.shape[1]}")
        
        return reduced_features, high_corr_features
    
    def variance_based_selection(self, feature_matrix, threshold=0.01):
        """Remove low-variance features"""
        print(f"=== VARIANCE-BASED FEATURE SELECTION (threshold={threshold}) ===")
        
        # Calculate variance for each feature
        feature_variances = feature_matrix.var()
        
        # Select features above threshold
        high_var_features = feature_variances[feature_variances > threshold].index
        low_var_features = feature_variances[feature_variances <= threshold].index
        
        print(f"Removing {len(low_var_features)} low-variance features")
        print(f"Keeping {len(high_var_features)} high-variance features")
        
        reduced_features = feature_matrix[high_var_features]
        
        return reduced_features, low_var_features.tolist()

class ModelOptimizer:
    """Advanced model optimization techniques"""
    
    def __init__(self):
        self.best_params = None
        self.optimization_history = []
    
    def hyperparameter_search(self, model_class, X_train, y_train, X_val, y_val):
        """Systematic hyperparameter optimization"""
        print("=== HYPERPARAMETER OPTIMIZATION ===")
        
        param_grid = {
            'learning_rate': [0.001, 0.0005, 0.0001],
            'batch_size': [16, 32, 64],
            'dropout_rate': [0.3, 0.5, 0.7],
            'l2_reg': [0.01, 0.001, 0.0001]
        }
        
        best_score = 0
        best_params = None
        
        from itertools import product
        
        # Grid search
        param_combinations = list(product(*param_grid.values()))
        
        for i, params in enumerate(param_combinations):
            param_dict = dict(zip(param_grid.keys(), params))
            print(f"\\nTesting params {i+1}/{len(param_combinations)}: {param_dict}")
            
            try:
                # Build model with current parameters
                model = model_class(
                    input_dim=X_train.shape[1],
                    num_classes=y_train.shape[1]
                )
                
                # Modify model architecture based on parameters
                model.build_model()
                model.compile_model(learning_rate=param_dict['learning_rate'])
                
                # Train for few epochs
                history = model.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=10,  # Quick evaluation
                    batch_size=param_dict['batch_size'],
                    verbose=0
                )
                
                # Evaluate
                val_loss = min(history.history['val_loss'])
                val_acc = max(history.history['val_binary_accuracy'])
                
                # Combined score (you can modify this)
                score = val_acc - 0.1 * val_loss
                
                self.optimization_history.append({
                    'params': param_dict,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = param_dict
                    print(f"  New best score: {score:.4f}")
                
            except Exception as e:
                print(f"  Error with params: {e}")
                continue
        
        self.best_params = best_params
        print(f"\\nBest parameters: {best_params}")
        print(f"Best score: {best_score:.4f}")
        
        return best_params
    
    def ensemble_prediction(self, models, X_test):
        """Ensemble multiple models for better predictions"""
        print("=== ENSEMBLE PREDICTION ===")
        
        predictions = []
        for i, model in enumerate(models):
            pred = model.predict(X_test)
            predictions.append(pred)
            print(f"Model {i+1} prediction shape: {pred.shape}")
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Weighted average (you can implement more sophisticated weighting)
        weights = np.ones(len(models)) / len(models)
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred, weighted_pred

class DataAugmentation:
    """Data augmentation techniques for imbalanced datasets"""
    
    def __init__(self):
        self.samplers = {}
    
    def apply_smote(self, X, y, sampling_strategy='auto'):
        """Apply SMOTE for minority class oversampling"""
        print("=== APPLYING SMOTE DATA AUGMENTATION ===")
        
        # Convert multilabel to multiclass for SMOTE
        if y.ndim > 1:
            # Use label combinations as classes
            y_combined = [''.join(map(str, row)) for row in y]
            unique_labels = list(set(y_combined))
            
            print(f"Original class distribution:")
            from collections import Counter
            print(Counter(y_combined))
            
            # Apply SMOTE only if we have enough samples
            if len(unique_labels) > 1 and min(Counter(y_combined).values()) >= 2:
                le = LabelEncoder()
                y_encoded = le.fit_transform(y_combined)
                
                smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
                
                # Convert back to multilabel
                y_resampled_combined = le.inverse_transform(y_resampled)
                y_resampled_multilabel = np.array([[int(c) for c in label] for label in y_resampled_combined])
                
                print(f"After SMOTE - Samples: {X_resampled.shape[0]}")
                print(f"New class distribution:")
                print(Counter(y_resampled_combined))
                
                return X_resampled, y_resampled_multilabel
            else:
                print("Not enough samples for SMOTE, returning original data")
                return X, y
        else:
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
            return smote.fit_resample(X, y)
    
    def noise_augmentation(self, X, noise_factor=0.01):
        """Add small amounts of noise to features"""
        print(f"=== APPLYING NOISE AUGMENTATION (factor={noise_factor}) ===")
        
        noise = np.random.normal(0, noise_factor, X.shape)
        X_augmented = X + noise
        
        # Ensure non-negative values (for abundance data)
        X_augmented = np.maximum(X_augmented, 0)
        
        return X_augmented

# Example usage script
def run_comprehensive_analysis():
    """Run comprehensive analysis on your gut microbiota data"""
    
    print("=== COMPREHENSIVE GUT MICROBIOTA ANALYSIS ===\\n")
    
    # Initialize analyzers
    analyzer = DataAnalyzer()
    feature_selector = AdvancedFeatureSelector()
    optimizer = ModelOptimizer()
    augmenter = DataAugmentation()
    
    # Load your data (modify file paths as needed)
    file_paths = [
        'COVID_19.csv', 'Anorexia.csv', 'Hypertension.csv',
        'Tuberculosis.csv', 'Melanoma.csv'
        # Add more files here
    ]
    
    try:
        from gutmlc_python_model import GutMLCPipeline  # Import your main model
        
        # Initialize main pipeline
        gutmlc = GutMLCPipeline()
        
        # Prepare data
        X, y, sample_info = gutmlc.prepare_data(file_paths)
        
        # Run comprehensive analysis
        print("\\n" + "="*50)
        print("STEP 1: DATA DISTRIBUTION ANALYSIS")
        print("="*50)
        
        # Analyze data distribution
        combined_df = pd.concat([pd.read_csv(f) for f in file_paths], ignore_index=True)
        stats = analyzer.analyze_dataset_distribution(combined_df)
        analyzer.plot_data_distribution(combined_df)
        
        print("\\n" + "="*50)
        print("STEP 2: MICROBIAL PATTERN ANALYSIS")
        print("="*50)
        
        # Analyze microbial patterns
        feature_matrix = pd.DataFrame(X)  # Convert to DataFrame for analysis
        patterns = analyzer.analyze_microbial_patterns(feature_matrix, sample_info)
        
        print("\\n" + "="*50)
        print("STEP 3: DIMENSIONALITY ANALYSIS")
        print("="*50)
        
        # Dimensionality reduction analysis
        dim_results = analyzer.perform_dimensionality_analysis(X, y, sample_info)
        
        print("\\n" + "="*50)
        print("STEP 4: ADVANCED FEATURE SELECTION")
        print("="*50)
        
        # Advanced feature selection
        feature_matrix_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Variance-based selection
        X_var_selected, low_var_features = feature_selector.variance_based_selection(feature_matrix_df)
        
        # Correlation-based selection
        X_final, high_corr_features = feature_selector.correlation_based_selection(X_var_selected)
        
        print("\\n" + "="*50)
        print("STEP 5: DATA AUGMENTATION")
        print("="*50)
        
        # Apply data augmentation for imbalanced classes
        X_augmented, y_augmented = augmenter.apply_smote(X_final.values, y)
        
        print("\\n" + "="*50)
        print("STEP 6: MODEL TRAINING WITH OPTIMIZATION")
        print("="*50)
        
        # Split augmented data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_augmented, y_augmented, test_size=0.2, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Train optimized model
        gutmlc_optimized = GutMLCPipeline()
        gutmlc_optimized.preprocessor = gutmlc.preprocessor  # Use same preprocessor
        
        # Train with optimized data
        history = gutmlc_optimized.train_model(
            X_train, y_train, 
            validation_split=0.0,  # We already have validation set
            epochs=100,
            batch_size=32
        )
        
        # Manual validation evaluation during training
        val_predictions = gutmlc_optimized.model.model.predict(X_val)
        val_loss = gutmlc_optimized.model.model.evaluate(X_val, y_val, verbose=0)
        
        print("\\n" + "="*50)
        print("STEP 7: FINAL EVALUATION")
        print("="*50)
        
        # Final evaluation
        results = gutmlc_optimized.evaluate_model(X_test, y_test)
        
        # Plot training history
        gutmlc_optimized.plot_training_history()
        
        print("\\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        
        return {
            'model': gutmlc_optimized,
            'results': results,
            'analysis_stats': stats,
            'feature_selection': {
                'removed_low_var': low_var_features,
                'removed_high_corr': high_corr_features,
                'final_features': X_final.shape[1]
            },
            'augmentation': {
                'original_samples': X.shape[0],
                'augmented_samples': X_augmented.shape[0]
            }
        }
        
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_comprehensive_analysis()