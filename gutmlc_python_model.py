"""
GutMLC: Advanced Multi-Label Classification for Disease Prediction from Gut Microbiota
Implementation based on the research paper with enhancements for your dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.metrics import hamming_loss, coverage_error, average_precision_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Data preprocessing pipeline for gut microbiota data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.svd = TruncatedSVD(n_components=100, random_state=42)
        self.feature_selector = None
        self.mlb = MultiLabelBinarizer()
        
    def load_and_combine_data(self, file_paths):
        """Load and combine multiple disease CSV files"""
        all_data = []
        
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
                print(f"Loaded {file_path}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if not all_data:
            raise ValueError("No data files could be loaded")
            
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined dataset: {len(combined_df)} total rows")
        return combined_df
    
    def create_sample_feature_matrix(self, df):
        """Create sample-feature matrix from long-format data"""
        print("Creating sample-feature matrix...")
        
        # Create pivot table: samples x organisms
        feature_matrix = df.pivot_table(
            index='run_id', 
            columns='scientific_name', 
            values='relative_abundance',
            fill_value=0,
            aggfunc='mean'  # Handle duplicates by averaging
        )
        
        # Create sample metadata
        sample_info = df.groupby('run_id').agg({
            'disease_name': 'first',
            'host_age': 'first',
            'sex': 'first',
            'country': 'first',
            'phenotype': 'first'
        }).reset_index()
        
        # Create multilabel target matrix
        # Group diseases by sample to handle multiple diseases per sample
        sample_diseases = df.groupby('run_id')['disease_name'].apply(list).reset_index()
        sample_diseases['disease_name'] = sample_diseases['disease_name'].apply(lambda x: list(set(x)))
        
        print(f"Feature matrix shape: {feature_matrix.shape}")
        print(f"Number of samples: {len(sample_info)}")
        print(f"Number of unique organisms: {feature_matrix.shape[1]}")
        
        return feature_matrix, sample_info, sample_diseases
    
    def preprocess_features(self, X, fit=True):
        """Preprocess feature matrix"""
        print("Preprocessing features...")
        
        # Handle missing values
        if fit:
            X_imputed = self.imputer.fit_transform(X)
        else:
            X_imputed = self.imputer.transform(X)
        
        # Apply log transformation to handle skewed abundance data
        X_log = np.log1p(X_imputed)
        
        # Standardize features
        if fit:
            X_scaled = self.scaler.fit_transform(X_log)
        else:
            X_scaled = self.scaler.transform(X_log)
        
        # Dimensionality reduction using SVD (matrix factorization as in paper)
        if fit:
            X_reduced = self.svd.fit_transform(X_scaled)
        else:
            X_reduced = self.svd.transform(X_scaled)
        
        print(f"Reduced feature matrix shape: {X_reduced.shape}")
        return X_reduced
    
    def create_multilabel_targets(self, sample_diseases):
        """Create multilabel binary matrix"""
        diseases_list = sample_diseases['disease_name'].tolist()
        
        # Fit and transform to binary matrix
        y_multilabel = self.mlb.fit_transform(diseases_list)
        
        print(f"Multilabel target shape: {y_multilabel.shape}")
        print(f"Number of unique diseases: {len(self.mlb.classes_)}")
        print(f"Disease classes: {self.mlb.classes_}")
        
        return y_multilabel

class SemanticSimilarity:
    """Compute semantic similarities as described in the paper"""
    
    def __init__(self):
        self.disease_similarity = None
        self.microbe_similarity = None
    
    def compute_disease_similarity(self, diseases):
        """Compute disease semantic similarity matrix"""
        # Simple implementation - can be enhanced with MeSH hierarchy
        n_diseases = len(diseases)
        similarity_matrix = np.eye(n_diseases)
        
        # Add some disease relationships (simplified)
        disease_groups = {
            'metabolic': ['Obesity', 'Diabetes', 'Hypertension'],
            'autoimmune': ['Rheumatoid Arthritis', 'Crohn Disease', 'Ulcerative Colitis'],
            'neurological': ['Alzheimer Disease', 'Autism Spectrum Disorder'],
            'cancer': ['Breast Neoplasms', 'Melanoma', 'Adenoma']
        }
        
        for group_diseases in disease_groups.values():
            for i, d1 in enumerate(diseases):
                for j, d2 in enumerate(diseases):
                    if d1 in group_diseases and d2 in group_diseases and i != j:
                        similarity_matrix[i, j] = 0.7
        
        self.disease_similarity = similarity_matrix
        return similarity_matrix
    
    def compute_microbe_similarity(self, microbes):
        """Compute microbe semantic similarity based on taxonomy"""
        n_microbes = len(microbes)
        similarity_matrix = np.eye(n_microbes)
        
        # Simple genus-level similarity
        for i, m1 in enumerate(microbes):
            for j, m2 in enumerate(microbes):
                if i != j:
                    # Extract genus (first word before space)
                    genus1 = m1.split()[0] if ' ' in m1 else m1
                    genus2 = m2.split()[0] if ' ' in m2 else m2
                    
                    if genus1 == genus2:
                        similarity_matrix[i, j] = 0.8
                    elif genus1.startswith(genus2[:3]) or genus2.startswith(genus1[:3]):
                        similarity_matrix[i, j] = 0.4
        
        self.microbe_similarity = similarity_matrix
        return similarity_matrix

class FocalLoss(keras.losses.Loss):
    """Focal Loss with Debiased Inverse Weighting (FL-DIW) from the paper"""
    
    def __init__(self, alpha=1.0, gamma=2.0, class_weights=None, name="focal_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def call(self, y_true, y_pred):
        # Ensure predictions are in valid range
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        
        # Calculate focal loss components
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        
        # Focal loss formula
        focal_loss = -alpha_t * tf.pow(1 - pt, self.gamma) * tf.math.log(pt)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            weights = tf.reduce_sum(self.class_weights * tf.cast(y_true, tf.float32), axis=-1, keepdims=True)
            focal_loss = focal_loss * weights
        
        return tf.reduce_mean(focal_loss)

class MultiLabelCNN:
    """Multi-Label CNN architecture based on the paper"""
    
    def __init__(self, input_dim, num_classes, embedding_dim=128):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.model = None
    
    def build_model(self):
        """Build the CNN model architecture"""
        inputs = layers.Input(shape=(self.input_dim,))
        
        # Reshape for 1D convolution
        x = layers.Reshape((self.input_dim, 1))(inputs)
        
        # Convolutional layers (VGG-like as mentioned in paper)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.5)(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer for multilabel classification
        outputs = layers.Dense(self.num_classes, activation='sigmoid')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def compile_model(self, learning_rate=0.001, class_weights=None):
        """Compile model with focal loss"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss = FocalLoss(alpha=0.25, gamma=2.0, class_weights=class_weights)
        
        metrics = [
            'binary_accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self.model

class GutMLCPipeline:
    """Complete GutMLC pipeline"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.semantic_sim = SemanticSimilarity()
        self.model = None
        self.history = None
        
    def prepare_data(self, file_paths):
        """Prepare data for training"""
        print("=== DATA PREPARATION ===")
        
        # Load and combine data
        df = self.preprocessor.load_and_combine_data(file_paths)
        
        # Create feature matrix and targets
        X_raw, sample_info, sample_diseases = self.preprocessor.create_sample_feature_matrix(df)
        
        # Preprocess features
        X = self.preprocessor.preprocess_features(X_raw, fit=True)
        
        # Create multilabel targets
        y = self.preprocessor.create_multilabel_targets(sample_diseases)
        
        print(f"Final dataset: X{X.shape}, y{y.shape}")
        return X, y, sample_info
    
    def train_model(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """Train the multilabel classification model"""
        print("=== MODEL TRAINING ===")
        
        # Calculate class weights for imbalanced data
        class_weights = []
        for i in range(y.shape[1]):
            pos_weight = np.sum(y[:, i] == 0) / np.sum(y[:, i] == 1)
            class_weights.append(min(pos_weight, 10.0))  # Cap weights
        class_weights = np.array(class_weights)
        
        # Build and compile model
        self.model = MultiLabelCNN(
            input_dim=X.shape[1], 
            num_classes=y.shape[1]
        )
        self.model.build_model()
        self.model.compile_model(class_weights=class_weights)
        
        print(self.model.model.summary())
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            ModelCheckpoint('best_gutmlc_model.h5', save_best_only=True, monitor='val_loss')
        ]
        
        # Train model
        self.history = self.model.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("=== MODEL EVALUATION ===")
        
        # Predictions
        y_pred = self.model.model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Multilabel metrics
        hamming = hamming_loss(y_test, y_pred_binary)
        coverage = coverage_error(y_test, y_pred)
        avg_precision = average_precision_score(y_test, y_pred, average='micro')
        
        print(f"Hamming Loss: {hamming:.4f}")
        print(f"Coverage Error: {coverage:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        
        # Per-class metrics
        disease_names = self.preprocessor.mlb.classes_
        for i, disease in enumerate(disease_names):
            if np.sum(y_test[:, i]) > 0:  # Only if disease is present in test set
                precision = np.sum((y_test[:, i] == 1) & (y_pred_binary[:, i] == 1)) / max(np.sum(y_pred_binary[:, i]), 1)
                recall = np.sum((y_test[:, i] == 1) & (y_pred_binary[:, i] == 1)) / np.sum(y_test[:, i])
                f1 = 2 * precision * recall / max(precision + recall, 1e-7)
                print(f"{disease}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        return {
            'hamming_loss': hamming,
            'coverage_error': coverage,
            'average_precision': avg_precision,
            'predictions': y_pred,
            'binary_predictions': y_pred_binary
        }
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.history.history['binary_accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_binary_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()

# Example usage and main execution
def main():
    """Main execution function"""
    
    # Initialize pipeline
    gutmlc = GutMLCPipeline()
    
    # File paths - UPDATE THESE WITH YOUR ACTUAL FILE PATHS
    file_paths = [
        'COVID_19.csv',
        'Anorexia.csv', 
        'Hypertension.csv',
        'Tuberculosis.csv',
        'Melanoma.csv',
        'Obesity.csv',
        'Alzheimer_Disease.csv',
        'Crohn_Disease.csv'
        # Add more file paths here
    ]
    
    try:
        # Prepare data
        X, y, sample_info = gutmlc.prepare_data(file_paths)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Train model
        history = gutmlc.train_model(X_train, y_train, epochs=50, batch_size=16)
        
        # Evaluate model
        results = gutmlc.evaluate_model(X_test, y_test)
        
        # Plot results
        gutmlc.plot_training_history()
        
        print("\\n=== TRAINING COMPLETED ===")
        print("Model saved as 'best_gutmlc_model.h5'")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()