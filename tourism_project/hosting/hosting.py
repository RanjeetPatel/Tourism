# ==========================================================
# Visit With Us
# Hugging Face Space Deployment
#
# This script uploads the deployment folder
# to the Hugging Face Space.
# ==========================================================

# ==========================================================
# Import Required Libraries
# ==========================================================

import os

from huggingface_hub import HfApi

# ==========================================================
# Configuration
# ==========================================================

REPO_ID = "ranjeetpatel29/Bank-Customer-Churn"

DEPLOYMENT_FOLDER = "tourism_project/deployment"

# ==========================================================
# Create Hugging Face API Client
# ==========================================================

token = os.getenv("HF_TOKEN")

api = HfApi(token=token)

# ==========================================================
# Upload Deployment Folder
# ==========================================================

def upload_space():

    print("=" * 80)
    print("Uploading Deployment Files to Hugging Face Space")
    print("=" * 80)

    api.upload_folder(

        folder_path=DEPLOYMENT_FOLDER,

        repo_id=REPO_ID,

        repo_type="space",

        path_in_repo="",

        ignore_patterns=[
            "__pycache__",
            "*.pyc",
            ".ipynb_checkpoints"
        ]

    )

    print("Deployment completed successfully.")

# ==========================================================
# Main Function
# ==========================================================

def main():

    upload_space()

# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print("\n")
        print("=" * 80)
        print("Deployment Failed")
        print("=" * 80)

        print(error)

        raise
