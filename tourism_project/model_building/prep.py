
# ==========================================================
# Visit With Us
# Data Preparation Pipeline
#
# Responsibilities
# 1. Load Raw Dataset from Hugging Face
# 2. Clean Dataset
# 3. Feature Engineering
# 4. Train/Test Split
# 5. Save Processed Dataset
# 6. Upload Processed Dataset
# ==========================================================

import os
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

from huggingface_hub import HfApi

warnings.filterwarnings("ignore")

# ==========================================================
# Hugging Face Configuration
# ==========================================================

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is not set.")

api = HfApi(token=HF_TOKEN)

REPO_ID = "ranjeetpatel29/Bank-Customer-Churn"
REPO_TYPE = "dataset"

DATASET_PATH = f"hf://datasets/{REPO_ID}/data/tourism.csv"

TARGET = "ProdTaken"

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print(f"Dataset Shape : {df.shape}")

# ==========================================================
# Data Cleaning
# ==========================================================

print("\nCleaning Dataset...")

drop_columns = [
    "Unnamed: 0",
    "CustomerID"
]

df.drop(
    columns=drop_columns,
    errors="ignore",
    inplace=True
)

df.drop_duplicates(inplace=True)

print(f"Dataset Shape After Cleaning : {df.shape}")

# ==========================================================
# Feature Engineering
# ==========================================================

print("\nCreating AgeGroup Feature...")

age_bins = [0, 25, 35, 45, 55, 100]

age_labels = [
    "Young Adult",
    "Adult",
    "Middle Age",
    "Senior Adult",
    "Senior Citizen"
]

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=age_bins,
    labels=age_labels,
    include_lowest=True,
    right=False
)

print(df["AgeGroup"].value_counts())

# ==========================================================
# Train Test Split
# ==========================================================

print("\nSplitting Dataset...")

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training Records : {len(X_train)}")
print(f"Testing Records  : {len(X_test)}")

# ==========================================================
# Save Processed Dataset
# ==========================================================

print("\nSaving Processed Dataset...")

train_df = X_train.copy()
train_df[TARGET] = y_train

test_df = X_test.copy()
test_df[TARGET] = y_test

train_df.to_csv(
    "train.csv",
    index=False
)

test_df.to_csv(
    "test.csv",
    index=False
)

print("Processed datasets saved successfully.")

# ==========================================================
# Upload Processed Dataset
# ==========================================================

print("\nUploading Processed Dataset...")

files = [
    "train.csv",
    "test.csv"
]

for file in files:

    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=f"processed/{file}",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE
    )

print("Processed datasets uploaded successfully.")

# ==========================================================
# Completed
# ==========================================================

print("\n" + "=" * 60)
print("Data Preparation Pipeline Completed Successfully")
print("=" * 60)
