# PARMS Usage Guide

Simple guide to run both analyses in the PARMS system.

## Setup (One Time)

```powershell
# Activate virtual environment
.\parms_env\Scripts\Activate.ps1
```

## Analysis 1: Parking Occupancy Prediction

**Dataset:** COCO-format parking space annotations  
**Location:** `data/pklot_reduced/`  
**Model:** Random Forest  

```bash
python parms.py
```

**Output:**
- `parms_model.pkl` - Trained model
- `parms_model_summary.txt` - Metrics summary
- `Generated Figures/parking_model_evaluation.png` - Evaluation plots

**Performance:** ~87% accuracy, ~88% F1-score

---

## Analysis 2: License Plate Recognition

**Dataset:** License plate images  
**Location:** `data/License Plates Dataset/`  
**Models:** 5 classifiers (Random Forest, Gradient Boosting, MLP, SVM, KNN)

```bash
python license_plate_analysis.py
```

**Output:**
- `license_plate_model.pkl` - Best model
- `license_plate_results.csv` - Model comparison table
- `license_plate_summary.txt` - Performance summary
- `Generated Figures/license_plate_analysis.png` - 6-panel comparison

**What it does:**
1. Loads all license plate images
2. Extracts features from images and plate numbers
3. Trains 5 different models
4. Compares performance
5. Saves best model
6. Generates visualization with:
   - Performance metrics comparison
   - Accuracy bar chart
   - F1-score comparison
   - Confusion matrix (best model)
   - Performance summary table
   - Model rankings

---

## Run Both Analyses

```powershell
# Activate environment
.\parms_env\Scripts\Activate.ps1

# Run parking occupancy analysis
python parms.py

# Run license plate analysis
python license_plate_analysis.py
```

## Notebook Analysis (Optional)

For deeper analysis with multiple models and visualizations:

```powershell
.\launch_notebook.bat
```

This opens `PARMS_Analysis.ipynb` with detailed parking occupancy analysis.

---

**That's it! Simple and clean.**
