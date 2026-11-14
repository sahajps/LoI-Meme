# Uploads our augmentation of both datasets to a Hugging Face dataset repo, with the same folders as here:
#   <repo>/README.md        # = Data/README_HF.md (the dataset card)
#   <repo>/FHM/ex/          # = Data/FHM/ex/
#   <repo>/MAMI/ex/         # = Data/MAMI/ex/
# The original datasets (ori/) are not uploaded. The HF token is read from Config/hf_token.txt and needs write access.
# A repo that doesn't exist yet is created as private. Files under FHM/ex/ and MAMI/ex/ on the hub that no longer
# exist here are deleted, so the repo always matches the local folders.
# Usage: python3 upload_data.py <HF repo link or id>, e.g., python3 upload_data.py https://huggingface.co/datasets/<user>/<name>
import os
import re
import sys
import pandas as pd
from huggingface_hub import HfApi

sys.path.append("../Config")
from config import DATA_DIR, DATASETS, HF_TOKEN_FILE, ex_dir

assert len(sys.argv) == 2, "usage: python3 upload_data.py <HF repo link or id>"
# https://huggingface.co/datasets/<user>/<name>[/tree/main] or <user>/<name> -> <user>/<name>
repo_id = re.sub(r"^(https?://)?(huggingface|hf)\.co/(datasets/)?", "", sys.argv[1].strip())
repo_id = "/".join(repo_id.split("/")[:2])

assert os.path.exists(HF_TOKEN_FILE), f"put an HF write token in {HF_TOKEN_FILE}"
hf_token = open(HF_TOKEN_FILE, "r", encoding="utf-8").read().strip()

# every file the dataset card points to must exist, and every new meme must have its image
for dataset in DATASETS:
    for split in ["train", "val"]:
        assert os.path.exists(f"{ex_dir(dataset)}ex_{split}_scale.jsonl"), f"{dataset}/ex/ex_{split}_scale.jsonl is missing"
        df = pd.read_json(f"{ex_dir(dataset)}ex_{split}.jsonl", lines=True, dtype=False)
        missing = [i for i in df.img if not os.path.exists(f"{ex_dir(dataset)}{i}")]
        assert not missing, f"{dataset}/ex/ex_{split}.jsonl: {len(missing)} images missing, e.g., {missing[:3]}"
        print(f"{dataset}/ex/ex_{split}.jsonl: {len(df)} new memes, all images found")

api = HfApi(token=hf_token)
api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

patterns = [f"{dataset}/ex/*" for dataset in DATASETS]
print(f"uploading {', '.join(patterns)} to {repo_id} ...", flush=True)
api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=DATA_DIR,
                  allow_patterns=patterns, delete_patterns=patterns, ignore_patterns=["*/.ipynb_checkpoints/*"],
                  commit_message=f"Upload {' and '.join(patterns)}")
api.upload_file(repo_id=repo_id, repo_type="dataset", path_or_fileobj=f"{DATA_DIR}/README_HF.md",
                path_in_repo="README.md", commit_message="Upload dataset card")
print(f"done: https://huggingface.co/datasets/{repo_id}")
