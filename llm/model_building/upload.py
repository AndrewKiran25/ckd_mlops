
# UPLOAD PROJECT TO HUGGING FACE HUB
import os
from huggingface_hub import (
    HfApi,
    create_repo,
    upload_folder
)

# CONFIGURATION
hf_token = os.getenv("HF_TOKEN")
REPO_ID = "Andrew2505/CKD-LLM"
PROJECT_FOLDER = "ckd_db"


# CHECK TOKEN
if not hf_token:
    raise ValueError("HF_TOKEN is missing!")

# Initialize API client
api = HfApi(token=hf_token)

print("Hugging Face API initialized.")


# CREATE REPOSITORY
create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
    exist_ok=True
)

print("Repository ready.")

# UPLOAD ENTIRE PROJECT FOLDER
api.upload_folder(
    folder_path=PROJECT_FOLDER,
    repo_id=REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
)

print("Upload completed successfully.")
