# visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import print_section

def plot_before_after(before, after, col, title):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(before[col], bins=30, kde=True, color="red")
    plt.title(f"Before: {title}")
    plt.subplot(1, 2, 2)
    sns.histplot(after[col], bins=30, kde=True, color="green")
    plt.title(f"After: {title}")
    plt.tight_layout()
    plt.show()

def visualize_all(df_raw, df_clean, df_final):
    print_section("VISUALIZATIONS")

    # 1. Missing values
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(df_raw.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title("Before Cleaning")
    plt.subplot(1, 2, 2)
    sns.heatmap(df_clean.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title("After Cleaning")
    plt.show()

    # 2. Occupancy by time of day
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_final, x="time_of_day", hue="category_id", palette="Set1")
    plt.title("Occupancy by Time of Day")
    plt.legend(["Empty", "Occupied"])
    plt.show()

    # 3. Weather impact
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df_final, x="weather", y="image_occupancy_rate", estimator="mean")
    plt.title("Avg Occupancy by Weather")
    plt.show()

    # 4. Augmentation effect
    plot_before_after(df_clean, df_final, "image_occupancy_rate", "Image Occupancy Rate")