# integration.py
import pandas as pd
from utils import print_section

def integrate_data(df: pd.DataFrame) -> pd.DataFrame:
    print_section("2. DATA INTEGRATION")

    # Occupancy rate per image
    occ_rate = (
        df.groupby("image_id")
        .apply(lambda g: g["category_id"].mean())  # 1=occupied
        .reset_index(name="image_occupancy_rate")
    )
    df = df.merge(occ_rate, on="image_id", how="left")

    # Add lot identifier from image path
    df["lot"] = df["filename"].str.split("/").str[-3]  # e.g., UFPR04

    print(f"Integrated: image_occupancy_rate, lot")
    return df
