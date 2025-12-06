"""
License Plate Character Recognition Analysis
Analyzes license plate first character recognition using complete preprocessing pipeline
Simulates OCR-like character group classification (more realistic than simple pattern matching)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from datetime import datetime
import re


class LicensePlateAnalyzer:
    """Complete license plate pattern analysis with preprocessing pipeline"""
    
    def __init__(self, data_dir="data/License Plates Dataset"):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.results = {}
        self.scaler = None
        
    def classify_first_character(self, plate_number):
        """Classify based on first character (more realistic OCR-like task)"""
        plate_clean = re.sub(r'[^A-Z0-9]', '', plate_number.upper())
        
        if not plate_clean:
            return 'Unknown'
        
        first_char = plate_clean[0]
        
        # Classify into character groups (simulates OCR confusion patterns)
        if first_char in 'ABCDEFGH':
            return 'Group_A-H'
        elif first_char in 'IJKLMNOP':
            return 'Group_I-P'
        elif first_char in 'QRSTUVWX':
            return 'Group_Q-X'
        elif first_char in 'YZ':
            return 'Group_Y-Z'
        elif first_char in '0123':
            return 'Group_0-3'
        elif first_char in '456789':
            return 'Group_4-9'
        else:
            return 'Unknown'
    
    def load_data(self):
        """STEP 1: Data Loading"""
        print("\n[STEP 1] DATA LOADING")
        print("="*60)
        
        data = []
        
        for split in ['train', 'val', 'test']:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                continue
                
            print(f"[INFO] Processing {split} split...")
            
            for img_path in split_dir.glob('*'):
                if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                    continue
                
                plate_number = img_path.stem.split('_')[0]
                
                try:
                    img = Image.open(img_path).convert('L')
                    
                    data.append({
                        'filename': img_path.name,
                        'plate_number': plate_number,
                        'split': split,
                        'image': img,
                        'original_size': img.size
                    })
                    
                except Exception as e:
                    continue
        
        df = pd.DataFrame(data)
        print(f"[OK] Loaded {len(df)} samples")
        print(f"[OK] Splits: {df['split'].value_counts().to_dict()}")
        
        return df
    
    def clean_data(self, df):
        """STEP 2: Data Cleaning"""
        print("\n[STEP 2] DATA CLEANING")
        print("="*60)
        
        original_count = len(df)
        
        df = df.drop_duplicates(subset=['plate_number'], keep='first')
        print(f"[OK] Removed {original_count - len(df)} duplicate plates")
        
        df = df[df['original_size'].apply(lambda s: s[0] * s[1] > 100)]
        df = df[df['plate_number'].str.len() > 0]
        print(f"[OK] Cleaned data: {len(df)} samples remaining")
        
        return df.reset_index(drop=True)
    
    def integrate_data(self, df):
        """STEP 3: Data Integration"""
        print("\n[STEP 3] DATA INTEGRATION")
        print("="*60)
        
        integrated_features = []
        
        for idx, row in df.iterrows():
            plate_number = row['plate_number']
            img = row['image']
            
            img_resized = img.resize((128, 64))
            img_array = np.array(img_resized)
            
            features = {
                'mean_intensity': np.mean(img_array),
                'std_intensity': np.std(img_array),
                'min_intensity': np.min(img_array),
                'max_intensity': np.max(img_array),
                'median_intensity': np.median(img_array),
                'intensity_range': np.max(img_array) - np.min(img_array),
                'plate_length': len(plate_number),
                'num_digits': sum(c.isdigit() for c in plate_number),
                'num_letters': sum(c.isalpha() for c in plate_number),
                'num_special': sum(not c.isalnum() for c in plate_number),
                'digit_ratio': sum(c.isdigit() for c in plate_number) / max(len(plate_number), 1),
                
                # Add more complex features for harder classification
                'intensity_variance': np.var(img_array),
                'edge_density': np.sum(np.abs(np.diff(img_array))) / img_array.size,
                
                'first_char_type': self.classify_first_character(plate_number)
            }
            
            integrated_features.append(features)
        
        features_df = pd.DataFrame(integrated_features)
        df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        
        print(f"[OK] Integrated {len(features_df.columns)} features")
        print(f"[OK] Character groups: {df['first_char_type'].value_counts().to_dict()}")
        
        return df
    
    def reduce_data(self, df):
        """STEP 4: Data Reduction"""
        print("\n[STEP 4] DATA REDUCTION")
        print("="*60)
        
        feature_cols = [
            'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
            'median_intensity', 'intensity_range', 'plate_length', 'num_digits',
            'num_letters', 'num_special', 'digit_ratio', 'intensity_variance', 'edge_density'
        ]
        
        print(f"[OK] Selected {len(feature_cols)} features")
        
        return df, feature_cols
    
    def transform_data(self, df, feature_cols):
        """STEP 5: Data Transformation"""
        print("\n[STEP 5] DATA TRANSFORMATION")
        print("="*60)
        
        X = df[feature_cols].values
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"[OK] Standardized {X_scaled.shape[1]} features")
        
        y = df['first_char_type'].values
        
        return X_scaled, y
    
    def discretize_data(self, df):
        """STEP 6: Data Discretization"""
        print("\n[STEP 6] DATA DISCRETIZATION")
        print("="*60)
        
        df['intensity_level'] = pd.cut(df['mean_intensity'], 
                                       bins=[0, 85, 170, 255], 
                                       labels=['Dark', 'Medium', 'Bright'])
        
        df['length_category'] = pd.cut(df['plate_length'], 
                                       bins=[0, 4, 7, 20], 
                                       labels=['Short', 'Medium', 'Long'])
        
        print(f"[OK] Created discrete categories")
        
        return df
    
    def augment_data(self, df):
        """STEP 7: Data Augmentation"""
        print("\n[STEP 7] DATA AUGMENTATION")
        print("="*60)
        print("[INFO] Augmentation techniques available:")
        print("  - Image rotation, brightness, contrast")
        print("[INFO] Using original data for this run")
        
        return df
    
    def split_data(self, X, y):
        """Split into train/test sets"""
        print("\n[INFO] TRAIN/TEST SPLIT")
        print("="*60)
        
        # Check if stratification is possible
        unique, counts = np.unique(y, return_counts=True)
        can_stratify = all(count >= 2 for count in counts)
        
        if can_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            print("[INFO] Some classes have <2 samples, stratification disabled")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        print(f"[OK] Train: {len(X_train)}, Test: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train 5 classification models"""
        print("\n[INFO] MODEL TRAINING (5 Models)")
        print("="*60)
        
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            ),
            'MLP Neural Network': MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
            ),
            'SVM': SVC(
                kernel='rbf', C=1.0, random_state=42
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5, n_jobs=-1
            )
        }
        
        results = []
        
        for name, model in models.items():
            print(f"\n[INFO] Training {name}...")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            print(f"[OK] Accuracy: {metrics['Accuracy']:.3f}, F1: {metrics['F1-Score']:.3f}")
            
            results.append(metrics)
            self.models[name] = model
        
        self.results = pd.DataFrame(results)
        return self.results, y_test, None
    
    def generate_visualizations(self, X_test, y_test):
        """Generate evaluation visualizations"""
        print("\n[INFO] GENERATING VISUALIZATIONS")
        print("="*60)
        
        os.makedirs("Generated Figures", exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('License Plate Character Recognition - Model Comparison', 
                    fontsize=16, fontweight='bold')
        
        # 1. Performance Metrics
        ax = axes[0, 0]
        metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(self.results))
        width = 0.2
        
        for i, metric in enumerate(metrics_cols):
            ax.bar(x + i*width, self.results[metric], width, label=metric)
        
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Performance Metrics Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(self.results['Model'], rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        # 2. Accuracy Ranking
        ax = axes[0, 1]
        sorted_results = self.results.sort_values('Accuracy', ascending=True)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_results)))
        bars = ax.barh(sorted_results['Model'], sorted_results['Accuracy'], color=colors)
        ax.set_xlabel('Accuracy', fontsize=11)
        ax.set_title('Model Accuracy Ranking', fontsize=13, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        for bar in bars:
            width_val = bar.get_width()
            ax.text(width_val, bar.get_y() + bar.get_height()/2, 
                   f'{width_val:.3f}', ha='left', va='center', fontsize=9)
        
        # 3. F1-Score Comparison
        ax = axes[0, 2]
        ax.bar(self.results['Model'], self.results['F1-Score'], 
              color=plt.cm.plasma(np.linspace(0.3, 0.9, len(self.results))))
        ax.set_ylabel('F1-Score', fontsize=11)
        ax.set_title('F1-Score by Model', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.set_xticklabels(self.results['Model'], rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Best Model Confusion Matrix
        ax = axes[1, 0]
        best_model_name = self.results.loc[self.results['Accuracy'].idxmax(), 'Model']
        best_model = self.models[best_model_name]
        y_pred = best_model.predict(X_test)
        
        unique_labels = np.unique(y_test)
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   xticklabels=unique_labels, yticklabels=unique_labels)
        ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        
        # 5. Performance Summary Table
        ax = axes[1, 1]
        ax.axis('off')
        
        table_data = []
        for _, row in self.results.iterrows():
            table_data.append([
                row['Model'][:20],
                f"{row['Accuracy']:.3f}",
                f"{row['F1-Score']:.3f}"
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Model', 'Accuracy', 'F1'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.5, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax.set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)
        
        # 6. Model Rankings
        ax = axes[1, 2]
        ax.axis('off')
        
        ranked = self.results.sort_values('Accuracy', ascending=False)
        summary_text = "Model Rankings\n" + "="*35 + "\n\n"
        
        for i, (_, row) in enumerate(ranked.iterrows(), 1):
            summary_text += f"{i}. {row['Model']}\n"
            summary_text += f"   Acc: {row['Accuracy']:.3f}\n"
            summary_text += f"   F1:  {row['F1-Score']:.3f}\n\n"
        
        summary_text += f"\nTest: {len(X_test)}\n"
        summary_text += f"Date: {datetime.now().strftime('%Y-%m-%d')}"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        output_path = "Generated Figures/license_plate_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved: {output_path}")
    
    def save_results(self):
        """Save models and results"""
        print("\n[INFO] SAVING RESULTS")
        print("="*60)
        
        best_idx = self.results['Accuracy'].idxmax()
        best_model_name = self.results.loc[best_idx, 'Model']
        best_model = self.models[best_model_name]
        
        model_data = {
            'model': best_model,
            'model_name': best_model_name,
            'scaler': self.scaler,
            'all_results': self.results.to_dict(),
            'training_date': datetime.now().isoformat()
        }
        
        with open('license_plate_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[OK] Model: license_plate_model.pkl ({best_model_name})")
        
        self.results.to_csv('license_plate_results.csv', index=False)
        print("[OK] Results: license_plate_results.csv")
        
        with open('license_plate_summary.txt', 'w') as f:
            f.write("License Plate Character Recognition Analysis\n")
            f.write("="*50 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Task: First Character Recognition (OCR-like)\n\n")
            f.write("Model Performance:\n")
            f.write("-"*50 + "\n")
            f.write(self.results.to_string(index=False))
            f.write(f"\n\nBest Model: {best_model_name}\n")
            f.write(f"Best Accuracy: {self.results['Accuracy'].max():.4f}\n")
        
        print("[OK] Summary: license_plate_summary.txt")
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*60)
        print("LICENSE PLATE CHARACTER RECOGNITION ANALYSIS")
        print("="*60)
        
        df = self.load_data()
        df = self.clean_data(df)
        df = self.integrate_data(df)
        df, feature_cols = self.reduce_data(df)
        X, y = self.transform_data(df, feature_cols)
        df = self.discretize_data(df)
        df = self.augment_data(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        results, y_test_final, _ = self.train_models(X_train, y_train, X_test, y_test)
        self.generate_visualizations(X_test, y_test)
        self.save_results()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"\nBest Model: {results.loc[results['Accuracy'].idxmax(), 'Model']}")
        print(f"Best Accuracy: {results['Accuracy'].max():.3f}")
        print(f"\nOutput Files:")
        print("  - license_plate_model.pkl")
        print("  - license_plate_results.csv")
        print("  - license_plate_summary.txt")
        print("  - Generated Figures/license_plate_analysis.png")
        print("\n" + "="*60 + "\n")


def main():
    """Main entry point"""
    analyzer = LicensePlateAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
