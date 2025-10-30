# cleaning.py
import pandas as pd
from utils import print_section

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print_section("1. DATA CLEANING")
    before = len(df)
    print(f"Before: {before} rows")

    # --- FIX: Convert bbox list → tuple so it's hashable ---
    df['bbox_tuple'] = df['bbox'].apply(tuple)

    # 1. Remove duplicates using image_id + bbox_tuple
    df = df.drop_duplicates(subset=["image_id", "bbox_tuple"])

    # 2. Remove invalid bboxes (area=0)
    df = df[df["area"] > 0]

    # 3. Remove missing datetime
    df = df.dropna(subset=["datetime"])

    # 4. Standardize weather names
    weather_map = {"Sunny": "Sunny", "Cloudy": "Cloudy", "Rainy": "Rainy", "Unknown": "Unknown"}
    df["weather"] = df["weather"].map(weather_map).fillna("Unknown")

    # Drop helper column
    df = df.drop(columns=["bbox_tuple"])

    after = len(df)
    print(f"After: {after} rows (removed {before - after})")
    return df