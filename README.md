# PARMS - Parking Management System

A simple parking occupancy prediction system using machine learning. Processes COCO-format parking datasets and trains models to predict parking space availability.

## Two Analysis Tools

### 1. Parking Occupancy Analysis (`parms.py`)
### 2. License Plate Recognition (`license_plate_analysis.py`)

## Quick Start

### Activate Environment
```powershell
.\parms_env\Scripts\Activate.ps1
```

### Run Parking Occupancy Analysis
```bash
python parms.py
```

**This command:**
- Processes full parking dataset
- Trains Random Forest model
- Generates evaluation plots and metrics
- Saves trained model (`parms_model.pkl`)

### Run License Plate Recognition Analysis
```bash
python license_plate_analysis.py
```

**This command:**
- Loads license plate images from `data/License Plates Dataset/`
- Trains 5 models: Random Forest, Gradient Boosting, MLP, SVM, K-Nearest Neighbors
- Generates comparison visualizations
- Saves best model (`license_plate_model.pkl`)

## Features

- **Two Complete Pipelines**: Parking occupancy + License plate recognition
- **Full Dataset**: Uses entire datasets (no artificial limits)
- **Clean Output**: Simple text markers, no emojis
- **Clear Visualizations**: Multi-panel evaluation reports
- **Multiple Models**: Compare 5 different ML algorithms for license plates

## Project Structure

```
parms_preprocessing/
├── parms.py              # Main application (CLI interface)
├── data_processor.py     # Data loading, cleaning, feature engineering  
├── model_trainer.py      # ML training, evaluation, visualization
├── config.py            # Configuration and utilities
├── requirements.txt     # Python dependencies
├── data/               # Dataset folder
│   ├── processed_pklot_parms_coco.csv  # Processed data (auto-generated)
│   └── pklot_reduced/  # Raw COCO dataset
│       ├── train/
│       ├── valid/
│       └── test/
└── Generated Figures/  # Output visualizations (auto-generated)
```

## Command Options

### Parking Occupancy (`parms.py`)
```bash
# Run everything (default - recommended)
python parms.py

# Data preprocessing only
python parms.py --preprocess

# Model training only  
python parms.py --train

# Interactive demo
python parms.py --demo
```

### License Plate Recognition (`license_plate_analysis.py`)
```bash
# Run complete analysis
python license_plate_analysis.py
```

## Output Files

### Parking Occupancy Analysis
- `parms_model.pkl` - Trained Random Forest model
- `parms_model_summary.txt` - Performance metrics and feature list
- `data/processed_pklot_parms_coco.csv` - Processed dataset
- `Generated Figures/parking_model_evaluation.png` - 4-panel evaluation report

### License Plate Recognition Analysis
- `license_plate_model.pkl` - Best trained model
- `license_plate_results.csv` - All model comparisons
- `license_plate_summary.txt` - Performance summary
- `Generated Figures/license_plate_analysis.png` - 6-panel comparison report

## Configuration

Key settings in `config.py`:
- `max_samples`: None (uses full dataset)
- `test_size`: 0.2 (20% test split)
- `random_seed`: 42

## Requirements

All dependencies are in `requirements.txt`:
- scikit-learn (Random Forest)
- pandas, numpy (data processing)
- matplotlib, seaborn (visualizations)
- Pillow (image handling)

## Expected Performance

### Parking Occupancy (PKLot dataset)
- **Accuracy**: ~87%
- **F1-Score**: ~88%
- **AUC**: ~95%
- **Training Time**: 2-5 minutes

### License Plate Recognition
- **5 Models Compared**: Random Forest, Gradient Boosting, MLP, SVM, KNN
- **Performance**: Varies by model (see generated report)
- **Training Time**: 5-10 minutes for all models

## Troubleshooting

**Data not found?**
- Ensure COCO dataset is in `data/pklot_reduced/` with train/valid/test folders
- Each folder needs `_annotations.coco.json`

**Import errors?**
- Activate environment: `.\parms_env\Scripts\Activate.ps1`
- Install dependencies: `pip install -r requirements.txt`

## How It Works

1. **Data Processing**: Loads COCO annotations, cleans data, engineers features (occupancy rates, time features, bbox coordinates)
2. **Model Training**: Random Forest classifier with 5-fold cross-validation
3. **Evaluation**: Generates confusion matrix, ROC curve, feature importance plots, and metrics
4. **Output**: Saves model and creates evaluation report

## Notebook Analysis

For deeper analysis, run `PARMS_Analysis.ipynb`:
- Compares multiple models (Random Forest, Gradient Boosting, MLP)
- Optional neural network models (if TensorFlow installed)
- Additional visualizations and metrics

Launch with: `.\launch_notebook.bat`

---

**PARMS**: Simple parking prediction for everyone.