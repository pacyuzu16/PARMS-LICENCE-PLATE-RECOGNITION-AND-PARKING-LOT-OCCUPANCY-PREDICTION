# main.py
from coco_loader import load_all_coco
from cleaning import clean_data
from integration import integrate_data
from reduction import reduce_data
from transformation import transform_data
from discretization import discretize_data
from augmentation import augment_data
from visualize import visualize_all
from utils import print_section

def main():
    print_section("PKLOT-640 COCO PREPROCESSING - GROUP 9: PARMS")

    # 0. Load
    raw_df = load_all_coco()
    df_before = raw_df.copy()

    # 1. Clean
    df = clean_data(raw_df)

    # 2. Integrate
    df = integrate_data(df)

    # 3. Reduce
    df = reduce_data(df)

    # 4. Transform
    df, le_w, le_l, scaler = transform_data(df)

    # 5. Discretize
    df = discretize_data(df)

    # 6. Augment
    final_df = augment_data(df)

    # Save
    final_df.to_csv("data/processed_pklot_parms_coco.csv", index=False)
    print(f"\nFinal dataset saved: {len(final_df)} rows")

    # Visualize
    visualize_all(df_before, df, final_df)

    print_section("PREPROCESSING DONE")

if __name__ == "__main__":
    main()