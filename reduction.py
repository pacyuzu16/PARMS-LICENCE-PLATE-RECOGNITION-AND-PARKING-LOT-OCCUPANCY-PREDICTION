# reduction.py
from utils import print_section

def reduce_data(df):
    print_section("3. DATA REDUCTION")
    before = df.shape[1]
    print(f"Before: {before} columns")

    # Drop raw bbox & filename
    df = df.drop(columns=["bbox", "filename", "area", "width", "height"], errors="ignore")

    after = df.shape[1]
    print(f"After: {after} columns (reduced by {before-after})")
    return df