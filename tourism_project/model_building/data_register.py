
import os

from google.colab import userdata
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Read HF token from Colab Secrets
token = userdata.get("HF_TOKEN")

api = HfApi(token=token)

# Reusing an existing Hugging Face repository because
# the free account currently cannot create another Docker Space.
repo_id = "ranjeetpatel29/Bank-Customer-Churn"
repo_type = "dataset"

folder_path = "tourism_project/data"

# Verify folder exists
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"{folder_path} does not exist.")

# Verify CSV files exist
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

if not csv_files:
    raise Exception("No CSV file found in tourism_project/data.")

print("Files to upload:")
for file in csv_files:
    print(f" - {file}")

# Create repository if it doesn't exist
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repository '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating repository '{repo_id}'...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print("Repository created successfully.")

# Upload dataset
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Dataset uploaded successfully.")
