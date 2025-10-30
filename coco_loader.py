# coco_loader.py
import json
import pandas as pd
from pathlib import Path
from utils import DATA_ROOT, print_section
from PIL import Image
import datetime

def load_coco_split(split: str) -> pd.DataFrame:
    """Load train/valid/test split from COCO JSON."""
    json_path = DATA_ROOT / split / "_annotations.coco.json"
    img_dir   = DATA_ROOT / split

    with open(json_path) as f:
        coco = json.load(f)

    # Build image lookup
    img_dict = {img["id"]: img for img in coco["images"]}

    records = []
    for ann in coco["annotations"]:
        img = img_dict[ann["image_id"]]
        records.append({
            "image_id": ann["image_id"],
            "filename": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "bbox": ann["bbox"],           # [x, y, w, h]
            "area": ann["area"],
            "category_id": ann["category_id"],
            "split": split
        })

    df = pd.DataFrame(records)
    # Add datetime from filename (e.g., 2013-04-15_17_40_12)
    df["datetime"] = pd.to_datetime(
        df["filename"].str.extract(r"(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})")[0],
        format="%Y-%m-%d_%H_%M_%S",
        errors="coerce"
    )
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour

    # Extract weather token (first part of filename) and map to canonical values
    df["weather"] = (
        df["filename"]
        .str.split("_")
        .str[0]
        .str.lower()
        .map({"sunny": "Sunny", "cloudy": "Cloudy", "rainy": "Rainy"})
        .fillna("Unknown")
    )

    return df

def load_all_coco() -> pd.DataFrame:
    print_section("LOADING PKLOT-640 COCO DATASET")
    df_train = load_coco_split("train")
    df_valid = load_coco_split("valid")
    df_test  = load_coco_split("test")
    df = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    print(f"Loaded {len(df)} bounding boxes from {len(df['image_id'].unique())} images.")
    return df