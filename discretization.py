# discretization.py
import pandas as pd
from utils import print_section

def discretize_data(df):
    print_section("5. DATA DISCRETIZATION")
    
    # Discretize hour into time-of-day
    bins = [0, 6, 12, 18, 24]
    labels = ["Night", "Morning", "Afternoon", "Evening"]
    df["time_of_day"] = pd.cut(df["hour"], bins=bins, labels=labels, include_lowest=True)

    print("Hour → time_of_day:")
    print(df["time_of_day"].value_counts())
    return df