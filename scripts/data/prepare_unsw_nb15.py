"""Minimal UNSW-NB15 prep: 1 file, 400K rows, float32."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

FEAT_CSV = Path("data/raw/UNSW-NB15/NUSW-NB15_features.csv")
DATA_CSV = Path("data/raw/UNSW-NB15/UNSW-NB15_1.csv")
DROP_COLS = ["srcip", "dstip", "sport", "dsport", "attack_cat"]

# Get column names
feat_df = pd.read_csv(FEAT_CSV, encoding="latin1")
feat_df.columns = feat_df.columns.str.strip()
col_names = feat_df.sort_values("No.")["Name"].str.strip().tolist()

keep_cols = [c for c in col_names if c not in DROP_COLS]
keep_idx = [col_names.index(c) for c in keep_cols]

print(f"Loading {DATA_CSV.name} (400K rows, {len(keep_cols)} cols)...")
df = pd.read_csv(DATA_CSV, header=None, usecols=keep_idx, nrows=400000, low_memory=False)
df.columns = keep_cols

# Convert numeric
for col in df.columns:
    if col != "Label":
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

# Map labels
df["Label"] = df["Label"].astype(str).str.strip()
df["Label"] = df["Label"].map({"0": "Normal", "0.0": "Normal", "1": "Attack", "1.0": "Attack"}).fillna("Normal")
df = df.rename(columns={"Label": "label"})  # Normalize to lowercase for rest of pipeline

print(f"Shape: {df.shape}")
print(f"Labels: {dict(df['label'].value_counts())}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.0f}MB")

# Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
train_df.to_csv("data/raw/unsw_nb15_train.csv", index=False)
test_df.to_csv("data/raw/unsw_nb15_test.csv", index=False)
print(f"Saved: train={len(train_df)}, test={len(test_df)}")
print("Done!")
