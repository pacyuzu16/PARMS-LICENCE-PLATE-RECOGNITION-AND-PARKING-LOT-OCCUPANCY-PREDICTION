# PARKING MANAGEMENT SYSTEM (PARMS) — Data Preprocessing

Group 9 | University of Rwanda — Computing Intelligence Module  
Lecturer: Dr. HITIMANA Eric | Date: 25/10/2025

Overview
--------
This repository demonstrates a complete data preprocessing pipeline for the PKLot-640 parking lot dataset, implementing six core preprocessing techniques and producing a cleaned CSV ready for downstream training.

Key features
- COCO loader and reduction to a balanced smaller dataset
- Cleaning, integration, column reduction, transformation, discretization and augmentation
- Visualizations for before/after inspection
- Final output: `data/processed_pklot_parms_coco.csv`

Quick links
- Run full pipeline: [parms_preprocessing/main.py](parms_preprocessing/main.py)
- Reduce dataset tool: [parms_preprocessing/reduce_dataset.py](parms_preprocessing/reduce_dataset.py)
- COCO loader: [parms_preprocessing/coco_loader.py](parms_preprocessing/coco_loader.py) (`coco_loader.load_all_coco`, `coco_loader.load_coco_split`)
- Steps implemented in code: `cleaning.clean_data`, `integration.integrate_data`, `reduction.reduce_data`, `transformation.transform_data`, `discretization.discretize_data`, `augmentation.augment_data`
- Helpers: [parms_preprocessing/utils.py](parms_preprocessing/utils.py) (`utils.DATA_ROOT`, `utils.print_section`)

Requirements
------------
- Python 3.8+
- pip packages:
  pandas, numpy, scikit-learn, matplotlib, seaborn, pillow

Install:
```sh
pip install pandas numpy scikit-learn matplotlib seaborn pillow
```

Dataset layout
--------------
Place your full PKLot COCO dataset here:
- data/pklot/train/
- data/pklot/valid/
- data/pklot/test/

Each split must contain:
- .jpg images
- _annotations.coco.json

Reducing the dataset (recommended)
----------------------------------
The reduction tool selects a balanced subset (default 500 images per split) and copies images + writes reduced COCO JSON to `data/pklot_reduced/`.

Run:
```sh
python reduce_dataset.py
```
Configuration:
- Change `IMAGES_PER_SPLIT` in [parms_preprocessing/reduce_dataset.py](parms_preprocessing/reduce_dataset.py) to adjust subset size.

Run full preprocessing
----------------------
The pipeline follows the sequence in [parms_preprocessing/main.py](parms_preprocessing/main.py):

0. Load COCO: `coco_loader.load_all_coco`  
1. Clean: `cleaning.clean_data`  
2. Integrate: `integration.integrate_data`  
3. Reduce columns: `reduction.reduce_data`  
4. Transform: `transformation.transform_data`  
5. Discretize: `discretization.discretize_data`  
6. Augment: `augmentation.augment_data`  
Visualize: `visualize.visualize_all`

Run:
```sh
python main.py
```
Output:
- CSV: `data/processed_pklot_parms_coco.csv`
- Visualizations: four interactive plots shown via matplotlib

Tips & troubleshooting
----------------------
- If your pipeline appears to use reduced data, check `DATA_ROOT` in [parms_preprocessing/utils.py](parms_preprocessing/utils.py).
- Missing datetime or parsing errors come from unexpected filename formats — inspect `filename` entries in your `_annotations.coco.json` (see [parms_preprocessing/coco_loader.py](parms_preprocessing/coco_loader.py)).
- If images are not found during reduction, reduce_dataset prints warnings with the missing path.

Extending / customizing
-----------------------
- Tweak augmentation strategy in [parms_preprocessing/augmentation.py](parms_preprocessing/augmentation.py).
- Adjust encoding/scaling in [parms_preprocessing/transformation.py](parms_preprocessing/transformation.py).
- Modify discretization bins in [parms_preprocessing/discretization.py](parms_preprocessing/discretization.py).

Citation
--------
Almeida, P. et al. (2015). _PKLot – A robust dataset for parking lot classification_. Expert Systems with Applications.

License
-------
owned by pacyuzu and pauline

Acknowledgements
----------------
This project uses the PKLot dataset (Roboflow public link: https://public.roboflow.ai/object-detection/pklot).


