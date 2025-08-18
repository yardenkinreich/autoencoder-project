import pandas as pd
from pathlib import Path

# Paths
robins_csv = "data/raw/lunar_crater_database_robbins_2018.csv"
daniels_dir = Path("/Users/yardenkinreich/Documents/Projects/Masters/daniel_crater_autoencoder/OneDrive_2025-08-03/Craters Classifier/craters_dataset")

# Read original Robins CSV
df = pd.read_csv(robins_csv)

# Get the list of Daniel's crater files (without extension)
daniel_files = {f.stem for f in daniels_dir.glob("*.jpeg")}

# Filter Robins CSV for craters that exist in Daniel's folder
filtered_df = df[df['CRATER_ID'].isin(daniel_files)]

# Save filtered CSV
filtered_csv_path = "data/raw/robins_filtered_for_daniel.csv"
filtered_df.to_csv(filtered_csv_path, index=False)

print(f"Filtered CSV saved to {filtered_csv_path}, containing {len(filtered_df)} craters.")
