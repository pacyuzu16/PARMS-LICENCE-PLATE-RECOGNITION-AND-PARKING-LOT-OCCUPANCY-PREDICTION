# transformation.py
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from utils import print_section

def transform_data(df):
    print_section("4. DATA TRANSFORMATION")
    df_t = df.copy()

    # Encode categorical
    le_weather = LabelEncoder()
    le_lot = LabelEncoder()
    df_t["weather_enc"] = le_weather.fit_transform(df_t["weather"])
    df_t["lot_enc"] = le_lot.fit_transform(df_t["lot"])

    # Scale occupancy rate
    scaler = MinMaxScaler()
    df_t["occ_scaled"] = scaler.fit_transform(df_t[["image_occupancy_rate"]])

    print("Applied: Label Encoding, Min-Max Scaling")
    return df_t, le_weather, le_lot, scaler