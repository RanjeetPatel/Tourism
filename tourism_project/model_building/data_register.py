
# ==========================================================
# Visit With Us
# Dataset Registration Pipeline
#
# Responsibilities
# 1. Verify Dataset Exists
# 2. Verify/Create Hugging Face Dataset Repository
# 3. Upload Raw Dataset
# 4. Verify Upload
# ==========================================================

import os

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ==========================================================
# Hugging Face Configuration
# ==========================================================

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is not set.")

api = HfApi(token=HF_TOKEN)

REPO_ID = "ranjeetpatel29/Bank-Customer-Churn"
REPO_TYPE = "dataset"

# ==========================================================
# Dataset Configuration
# ==========================================================

DATASET_FILE = "tourism_project/data/tourism.csv"
HF_DATASET_PATH = "data/tourism.csv"

# ==========================================================
# Verify Dataset
# ==========================================================

print("=" * 60)
print("Verifying Dataset...")
print("=" * 60)

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_FILE}"
    )

print(f"Dataset Found : {DATASET_FILE}")

# ==========================================================
# Verify Repository
# ==========================================================

print("\nChecking Hugging Face Dataset Repository...")

try:
    api.repo_info(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE
    )

    print(f"Repository '{REPO_ID}' already exists.")

except RepositoryNotFoundError:

    print(f"Creating repository '{REPO_ID}'...")

    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=False
    )

    print("Repository created successfully.")

# ==========================================================
# Upload Dataset
# ==========================================================

print("\nUploading Dataset...")

api.upload_file(
    path_or_fileobj=DATASET_FILE,
    path_in_repo=HF_DATASET_PATH,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE
)

print("Dataset uploaded successfully.")

# ==========================================================
# Verify Upload
# ==========================================================

print("\nVerifying Upload...")

files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE
)

if HF_DATASET_PATH in files:
    print("Upload verified successfully.")
else:
    raise Exception("Dataset upload verification failed.")

# ==========================================================
# Completed
# ==========================================================

print("\n" + "=" * 60)
print("Dataset Registration Completed Successfully")
print("=" * 60)
