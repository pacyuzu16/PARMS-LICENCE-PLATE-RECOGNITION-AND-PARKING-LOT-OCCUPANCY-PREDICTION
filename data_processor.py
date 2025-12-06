"""
PARMS Data Processor
Handles all data loading, cleaning, and preprocessing for parking datasets.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime
import os


class DataProcessor:
    """Handles all data processing operations."""
    
    def __init__(self, config):
        self.config = config
        self.data_root = Path(config.data_root)
        
    def process_all(self):
        """Run complete data processing pipeline."""
        print("Starting Data Processing Pipeline")
        print("-" * 40)
        
        # Step 1: Load raw data
        print("1. Loading COCO dataset...")
        raw_df = self._load_all_coco()
        
        # Step 2: Clean data
        print("2. Cleaning data...")
        df = self._clean_data(raw_df)
        
        # Step 3: Feature engineering
        print("3. Engineering features...")
        df = self._engineer_features(df)
        
        # Step 4: Transform data
        print("4. Transforming data...")
        df = self._transform_data(df)
        
        # Step 5: Final processing
        print("5. Final processing...")
        df = self._final_processing(df)
        
        # Step 6: Save processed data
        print("6. Saving processed data...")
        self._save_processed_data(df)
        
        print(f"Processing complete! Final dataset: {len(df):,} samples")
        return df
    
    def _load_all_coco(self):
        """Load all COCO dataset splits."""
        all_splits = []
        
        for split in ['train', 'valid', 'test']:
            split_df = self._load_coco_split(split)
            all_splits.append(split_df)
        
        df = pd.concat(all_splits, ignore_index=True)
        print(f"   Loaded {len(df):,} bounding boxes from {len(df['image_id'].unique()):,} images")
        return df
    
    def _load_coco_split(self, split):
        """Load single COCO split."""
        json_path = self.data_root / split / "_annotations.coco.json"
        
        if not json_path.exists():
            raise FileNotFoundError(f"COCO file not found: {json_path}")
        
        with open(json_path) as f:
            coco = json.load(f)
        
        # Build image lookup
        img_dict = {img["id"]: img for img in coco["images"]}
        
        # Extract records
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
        
        # Extract datetime from filename (e.g., 2013-04-15_17_40_12)
        df["datetime"] = pd.to_datetime(
            df["filename"].str.extract(r"(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})")[0],
            format="%Y-%m-%d_%H_%M_%S",
            errors="coerce"
        )
        df["date"] = df["datetime"].dt.date
        df["hour"] = df["datetime"].dt.hour
        
        # Extract weather from filename
        df["weather"] = (
            df["filename"]
            .str.split("_")
            .str[0]
            .str.lower()
            .map({"sunny": "Sunny", "cloudy": "Cloudy", "rainy": "Rainy"})
            .fillna("Unknown")
        )
        
        return df
    
    def _clean_data(self, df):
        """Clean and validate data."""
        before = len(df)
        
        # Convert bbox to tuple for duplicate detection
        df['bbox_tuple'] = df['bbox'].apply(tuple)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["image_id", "bbox_tuple"])
        
        # Remove invalid bboxes
        df = df[df["area"] > 0]
        
        # Remove missing datetime
        df = df.dropna(subset=["datetime"])
        
        # Standardize weather
        weather_map = {"Sunny": "Sunny", "Cloudy": "Cloudy", "Rainy": "Rainy", "Unknown": "Unknown"}
        df["weather"] = df["weather"].map(weather_map).fillna("Unknown")
        
        # Clean up
        df = df.drop(columns=["bbox_tuple"])
        
        after = len(df)
        print(f"   Cleaned: {before:,} → {after:,} samples (removed {before - after:,})")
        return df
    
    def _engineer_features(self, df):
        """Create new features from existing data."""
        
        # Image-level occupancy rate
        occ_rate = (
            df.groupby("image_id")["category_id"]
            .apply(lambda x: (x == 2).mean())  # category_id=2 means occupied
            .reset_index(name="image_occupancy_rate")
        )
        df = df.merge(occ_rate, on="image_id", how="left")
        
        # Extract parking lot identifier
        df["lot"] = df["filename"].str.split("/").str[-3]  # e.g., UFPR04
        
        # Bbox features
        df["bbox_x"] = df["bbox"].apply(lambda x: x[0])
        df["bbox_y"] = df["bbox"].apply(lambda x: x[1])
        df["bbox_w"] = df["bbox"].apply(lambda x: x[2])
        df["bbox_h"] = df["bbox"].apply(lambda x: x[3])
        
        # Time features
        df["is_weekend"] = df["datetime"].dt.dayofweek >= 5
        df["time_of_day"] = pd.cut(
            df["hour"], 
            bins=[0, 6, 12, 18, 24], 
            labels=["Night", "Morning", "Afternoon", "Evening"],
            include_lowest=True
        )
        
        # Image aspect ratio
        df["aspect_ratio"] = df["width"] / df["height"]
        
        print(f"   Engineered features: occupancy_rate, lot, bbox_coords, time_features")
        return df
    
    def _transform_data(self, df):
        """Apply transformations to features."""
        df_t = df.copy()
        
        # Encode categorical variables
        le_weather = LabelEncoder()
        le_lot = LabelEncoder()
        le_time = LabelEncoder()
        
        df_t["weather_enc"] = le_weather.fit_transform(df_t["weather"])
        df_t["lot_enc"] = le_lot.fit_transform(df_t["lot"])
        df_t["time_of_day_enc"] = le_time.fit_transform(df_t["time_of_day"])
        
        # Scale numerical features
        scaler = MinMaxScaler()
        numerical_cols = ["image_occupancy_rate", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "aspect_ratio"]
        
        for col in numerical_cols:
            if col in df_t.columns:
                df_t[f"{col}_scaled"] = scaler.fit_transform(df_t[[col]])
        
        print(f"   Transformed: categorical encoding, numerical scaling")
        return df_t
    
    def _final_processing(self, df):
        """Final processing and cleanup."""
        
        # Discretize continuous variables
        df["occupancy_level"] = pd.cut(
            df["image_occupancy_rate"],
            bins=[0, 0.3, 0.7, 1.0],
            labels=["Low", "Medium", "High"]
        )
        
        # Create binary target 
        df["is_occupied"] = (df["category_id"] == 2).astype(int)
        
        # Reduce dataset size if too large
        if len(df) > self.config.max_samples:
            print(f"   Dataset too large ({len(df):,}), sampling {self.config.max_samples:,} records")
            df = df.sample(n=self.config.max_samples, random_state=42)
        
        print(f"   Final processing: discretization, sampling")
        return df
    
    def _save_processed_data(self, df):
        """Save processed dataset."""
        os.makedirs(os.path.dirname(self.config.processed_data_path), exist_ok=True)
        df.to_csv(self.config.processed_data_path, index=False)
        
        # Save feature info
        feature_info = {
            "total_samples": len(df),
            "features": list(df.columns),
            "target_distribution": df["is_occupied"].value_counts().to_dict(),
            "processed_date": datetime.now().isoformat()
        }
        
        info_path = self.config.processed_data_path.replace(".csv", "_info.json")
        with open(info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"   Saved: {self.config.processed_data_path}")
        print(f"   Info: {info_path}")


def quick_load_processed_data(config):
    """Utility function to quickly load processed data."""
    if not os.path.exists(config.processed_data_path):
        raise FileNotFoundError(f"Processed data not found: {config.processed_data_path}")
    
    df = pd.read_csv(config.processed_data_path)
    print(f"Loaded processed data: {len(df):,} samples")
    return df