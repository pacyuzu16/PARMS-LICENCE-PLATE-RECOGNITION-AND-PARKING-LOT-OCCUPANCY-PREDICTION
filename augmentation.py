# augmentation.py
import pandas as pd
import numpy as np
from utils import print_section

def augment_data(df):
    print_section("6. DATA AUGMENTATION")
    before = len(df)
    print(f"Before: {before} samples")

    # 1. Add Gaussian noise to occupancy rate
    df_noisy = df.copy()
    noise = np.random.normal(0, 0.03, size=len(df_noisy))
    df_noisy["image_occupancy_rate"] = np.clip(df_noisy["image_occupancy_rate"] + noise, 0, 1)
    df_noisy["occ_scaled"] = (df_noisy["image_occupancy_rate"] - df_noisy["image_occupancy_rate"].mean()) / df_noisy["image_occupancy_rate"].std()

    # 2. Oversample rare occupied slots in Night
    rare = df_noisy[(df_noisy["category_id"] == 1) & (df_noisy["time_of_day"] == "Night")]
    df_aug = pd.concat([df_noisy, rare.sample(frac=2.0, replace=True, random_state=42)], ignore_index=True)

    after = len(df_aug)
    print(f"After: {after} samples (+{after-before})")
    return df_aug