"""
PARMS Model Trainer
Handles machine learning model training, evaluation, and prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from data_processor import quick_load_processed_data


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
    def train_and_evaluate(self):
        """Complete model training and evaluation pipeline."""
        print("Starting Model Training Pipeline")
        print("-" * 40)
        
        # Load data
        print("1. Loading processed data...")
        df = quick_load_processed_data(self.config)
        
        # Prepare features
        print("2. Preparing features...")
        X, y, feature_names = self._prepare_features(df)
        
        # Split data
        print("3. Splitting data...")
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        
        # Train model
        print("4. Training model...")
        model = self._train_model(X_train, y_train)
        
        # Evaluate model
        print("5. Evaluating model...")
        metrics = self._evaluate_model(model, X_test, y_test, feature_names)
        
        # Save model
        print("6. Saving model...")
        self._save_model(model, feature_names, metrics)
        
        print("Model training complete!")
        return model, metrics
    
    def _prepare_features(self, df):
        """Prepare features for training."""
        
        # Select features (exclude non-predictive columns)
        exclude_cols = [
            'image_id', 'filename', 'datetime', 'date', 'split', 'bbox',
            'category_id', 'is_occupied', 'weather', 'lot', 'time_of_day',
            'occupancy_level'  # These are categorical/text columns
        ]
        
        # Get numerical features
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and 
                       df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        # Ensure we have some features
        if not feature_cols:
            # Fallback: use basic available features
            basic_features = ['hour', 'image_occupancy_rate', 'weather_enc', 'lot_enc']
            feature_cols = [col for col in basic_features if col in df.columns]
        
        print(f"   Selected {len(feature_cols)} features: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")
        
        # Prepare X and y
        X = df[feature_cols].fillna(0)
        # Create binary target from category_id: 1=Occupied, 0=Empty
        y = (df['category_id'] == 2).astype(int)  # category_id=2 means occupied
        
        print(f"   Dataset shape: {X.shape}")
        print(f"   Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_cols
    
    def _split_data(self, X, y):
        """Split data into train/test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        print(f"   Training set: {X_train.shape[0]:,} samples")
        print(f"   Test set: {X_test.shape[0]:,} samples")
        
        return X_train, X_test, y_train, y_test
    
    def _train_model(self, X_train, y_train):
        """Train the parking occupancy model."""
        
        # Use Random Forest (fast and effective)
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all cores
        )
        
        print(f"   Training Random Forest...")
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        print(f"   Cross-validation F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return model
    
    def _evaluate_model(self, model, X_test, y_test, feature_names):
        """Comprehensive model evaluation."""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Print metrics
        print(f"   Model Performance:")
        for metric_name, value in metrics.items():
            print(f"      {metric_name.capitalize()}: {value:.3f}")
        
        # Generate visualizations
        self._create_evaluation_plots(y_test, y_pred, y_pred_proba, model, feature_names, metrics)
        
        return metrics
    
    def _create_evaluation_plots(self, y_test, y_pred, y_pred_proba, model, feature_names, metrics):
        """Create evaluation visualizations."""
        
        # Create figures directory
        fig_dir = "Generated Figures"
        os.makedirs(fig_dir, exist_ok=True)
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PARMS Model Evaluation Report', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        axes[0,0].set_xticklabels(['Empty', 'Occupied'])
        axes[0,0].set_yticklabels(['Empty', 'Occupied'])
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                      label=f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,1].set_xlim([0.0, 1.0])
        axes[0,1].set_ylim([0.0, 1.05])
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend(loc="lower right")
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Feature Importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10
            
            axes[1,0].barh(range(len(indices)), importances[indices])
            axes[1,0].set_yticks(range(len(indices)))
            axes[1,0].set_yticklabels([feature_names[i] for i in indices])
            axes[1,0].set_xlabel('Importance')
            axes[1,0].set_title('Top 10 Feature Importances')
            axes[1,0].invert_yaxis()
        
        # 4. Performance Summary
        axes[1,1].axis('off')
        summary_text = f"""
Model Performance Summary

Accuracy:  {metrics['accuracy']:.3f}
Precision: {metrics['precision']:.3f}  
Recall:    {metrics['recall']:.3f}
F1-Score:  {metrics['f1']:.3f}
AUC:       {roc_auc:.3f}

Dataset Info:
Total samples: {len(y_test):,}
Empty spots:   {sum(y_test == 0):,}
Occupied:      {sum(y_test == 1):,}

Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        axes[1,1].text(0.1, 0.9, summary_text, transform=axes[1,1].transAxes,
                      fontsize=11, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plot_path = os.path.join(fig_dir, "parking_model_evaluation.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Evaluation plot saved: {plot_path}")
    
    def _save_model(self, model, feature_names, metrics):
        """Save trained model and metadata."""
        
        # Save model
        model_data = {
            'model': model,
            'feature_names': feature_names,
            'metrics': metrics,
            'training_date': datetime.now().isoformat(),
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        }
        
        with open(self.config.model_save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"   Model saved: {self.config.model_save_path}")
        
        # Also save a human-readable summary
        summary_path = self.config.model_save_path.replace('.pkl', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("PARMS Model Training Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Type: {type(model).__name__}\n")
            f.write(f"Features: {len(feature_names)}\n\n")
            f.write("Performance Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric.capitalize()}: {value:.4f}\n")
            f.write(f"\nFeature Names:\n")
            for i, name in enumerate(feature_names, 1):
                f.write(f"  {i:2d}. {name}\n")
        
        print(f"   Summary saved: {summary_path}")
    
    def load_trained_model(self):
        """Load a previously trained model."""
        if not os.path.exists(self.config.model_save_path):
            raise FileNotFoundError(f"Model not found: {self.config.model_save_path}")
        
        with open(self.config.model_save_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        print(f"Loaded trained model: {self.config.model_save_path}")
        return self.model, model_data.get('metrics', {})
    
    def predict(self, X):
        """Make predictions with trained model."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_and_evaluate() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_and_evaluate() first.")
        
        return self.model.predict_proba(X)