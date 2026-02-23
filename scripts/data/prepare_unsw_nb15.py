"""
UNSW-NB15 Dataset Preparation Script.

Reads the 4 raw CSVs (no header), injects column names from the features CSV,
maps binary labels (0=Normal, 1=Attack), drops non-feature identifier columns,
and saves train/test splits to data/raw/.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR    = Path("data/raw/UNSW-NB15")
FEAT_CSV   = RAW_DIR / "NUSW-NB15_features.csv"
DATA_FILES = [RAW_DIR / f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
OUT_DIR    = Path("data/raw")

# Columns that are identifiers, not features (IP addresses, raw ports)
DROP_COLS  = ["srcip", "dstip", "sport", "dsport"]

# ── Load column names ──────────────────────────────────────────────────────────
def get_column_names() -> list:
    feat_df = pd.read_csv(FEAT_CSV, encoding="latin1")
    feat_df.columns = feat_df.columns.str.strip()
    # Column names are in the 'Name' column, ordered by 'No.'
    names = feat_df.sort_values("No.")["Name"].str.strip().tolist()
    return names


# ── Merge all 4 CSV files ─────────────────────────────────────────────────────
def load_and_merge(col_names: list) -> pd.DataFrame:
    chunks = []
    for f in DATA_FILES:
        print(f"  Loading {f.name} ...", end=" ", flush=True)
        df = pd.read_csv(f, header=None, names=col_names, low_memory=False)
        print(f"{len(df):,} rows")
        chunks.append(df)
    return pd.concat(chunks, ignore_index=True)


# ── Label mapping ──────────────────────────────────────────────────────────────
def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    label_col = "label"
    # The features CSV uses 'Label' (capital), try both
    if "Label" in df.columns and label_col not in df.columns:
        df = df.rename(columns={"Label": label_col})

    df[label_col] = df[label_col].astype(str).str.strip()
    df[label_col] = df[label_col].map({"0": "Normal", "1": "Attack"}).fillna("Normal")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n=== UNSW-NB15 Dataset Preparation ===\n")

    print("Step 1: Reading column names from features CSV...")
    col_names = get_column_names()
    print(f"  Found {len(col_names)} columns: {col_names[:5]} ... {col_names[-3:]}")

    print("\nStep 2: Loading and merging all 4 CSV files...")
    df = load_and_merge(col_names)
    print(f"  Total rows after merge: {len(df):,}")

    print("\nStep 3: Mapping binary labels to class names...")
    df = map_labels(df)
    print("  Label distribution:")
    for label, count in df["label"].value_counts().items():
        print(f"    {label}: {count:,}")

    print(f"\nStep 4: Dropping non-feature identifier columns: {DROP_COLS}")
    drop_actual = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_actual)
    print(f"  Remaining columns: {df.shape[1]}")

    print("\nStep 5: Removing duplicates...")
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {before - len(df):,} duplicates. Rows remaining: {len(df):,}")

    print("\nStep 6: Train/test split (80/20, stratified)...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    print("\nStep 7: Saving to data/raw/...")
    train_path = OUT_DIR / "unsw_nb15_train.csv"
    test_path  = OUT_DIR / "unsw_nb15_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"  Saved: {train_path}")
    print(f"  Saved: {test_path}")

    print("\n=== Done! ===\n")
    print("Next step — run training:")
    print("  python scripts/train.py --config configs/training/unsw_nb15.yaml\n")


if __name__ == "__main__":
    main()
