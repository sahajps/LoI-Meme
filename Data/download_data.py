# Downloads FHM and MAMI and writes both in a common format:
#   Data/<DATASET>/ori/img/*                      # images
#   Data/<DATASET>/ori/{train,val,test}.jsonl     # columns: id, img, text, label (img is relative to Data/<DATASET>/ori/)
#   Data/<DATASET>/ori/raw/                       # untouched original annotation files
# Usage: python3 download_data.py [FHM] [MAMI]
import os
import sys
import shutil
import zipfile
import pandas as pd
import kagglehub
from huggingface_hub import snapshot_download

sys.path.append("../Config")
from config import HF_TOKEN_FILE

datasets = sys.argv[1:] if len(sys.argv) > 1 else ["FHM", "MAMI"]

# HF token (Config/hf_token.txt) raises the HF rate limits; without it the download is anonymous
hf_token = open(HF_TOKEN_FILE, "r", encoding="utf-8").read().strip() if os.path.exists(HF_TOKEN_FILE) else ""
hf_token = hf_token or None


def save_split(df, path):
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def check_and_save(df, out_dir, split):
    # every meme must have a unique id and an existing image
    assert df.id.is_unique, f"{out_dir}/{split}: duplicate ids"
    missing = [i for i in df.img if not os.path.exists(f"{out_dir}/{i}")]
    if missing:
        print(f"[warn] {out_dir}/{split}: {len(missing)} images missing, dropping them, e.g., {missing[:3]}")
        df = df[~df.img.isin(missing)]
    save_split(df, f"{out_dir}/{split}.jsonl")
    print(f"{out_dir}/{split}.jsonl: {len(df)} memes, labels: {df.label.value_counts().to_dict()}")


# HF: https://huggingface.co/datasets/willgrobots/hateful_memes_zipped
# Official Phase 2 release: train, dev_seen, dev_unseen, test_seen, test_unseen (all labeled) + all 12,140 images in img.zip
def download_fhm():
    out_dir, raw_dir = "FHM/ori", "FHM/ori/raw"
    if hf_token is None:
        print(f"[hf] no token in {HF_TOKEN_FILE}, downloading anonymously", flush=True)
    if not os.path.exists(f"{out_dir}/img"):
        snapshot_download(repo_id="willgrobots/hateful_memes_zipped", repo_type="dataset", local_dir=raw_dir, token=hf_token)
        zipfile.ZipFile(f"{raw_dir}/img.zip").extractall(out_dir)  # zip entries are img/<id>.png
        os.remove(f"{raw_dir}/img.zip")
    else:  # images already extracted, only fetch annotations
        snapshot_download(repo_id="willgrobots/hateful_memes_zipped", repo_type="dataset", local_dir=raw_dir, allow_patterns=["*.jsonl"], token=hf_token)

    def read(f):
        return pd.read_json(f"{raw_dir}/{f}.jsonl", lines=True, dtype={"id": str})[["id", "img", "text", "label"]]

    train = read("train")
    # dev_seen and dev_unseen share 400 memes (see fhm_overlap_analysis.ipynb): val = dev_seen U dev_unseen
    val = pd.concat([read("dev_seen"), read("dev_unseen")], ignore_index=True).drop_duplicates("id")
    test = pd.concat([read("test_seen"), read("test_unseen")], ignore_index=True)
    assert not set(train.id) & set(val.id) and not set(test.id) & (set(train.id) | set(val.id)), "official splits should not overlap"

    for split, df in zip(["train", "val", "test"], [train, val, test]):
        check_and_save(df, out_dir, split)


# Kaggle: https://www.kaggle.com/datasets/nrizwan/mami-dataset (MAMI, SemEval-2022 Task 5)
# train.tsv + validation.tsv split the official 10,000 training memes; test.tsv is the official test set (all labeled)
def download_mami():
    out_dir, raw_dir = "MAMI/ori", "MAMI/ori/raw"
    os.makedirs(raw_dir, exist_ok=True)
    src = None
    if not os.path.exists(f"{out_dir}/img"):
        os.environ["KAGGLEHUB_CACHE"] = os.path.abspath(f"{raw_dir}/kagglehub")
        src = kagglehub.dataset_download("nrizwan/mami-dataset") + "/MAMI_Dataset/data/"
        for f in ["train.tsv", "validation.tsv", "test.tsv"]:
            shutil.copy(src + f, raw_dir)

    splits = {s: pd.read_csv(f"{raw_dir}/{f}.tsv", sep="\t") for s, f in [("train", "train"), ("val", "validation"), ("test", "test")]}
    # the training and test image folders share a few file names (see mami_overlap_analysis.ipynb):
    # those get a split suffix (e.g., 15001_val.jpg, 15001_t.jpg); all other images keep their original name
    shared = (set(splits["train"].file_name) | set(splits["val"].file_name)) & set(splits["test"].file_name)
    suffix = {"train": "tr", "val": "val", "test": "t"}

    def img_name(f, split):
        stem, ext = os.path.splitext(f)
        return f"{stem}_{suffix[split]}{ext}" if f in shared else f

    if src is not None:  # move all images into one folder
        os.makedirs(f"{out_dir}/img")
        for split, df in splits.items():
            folder = "test_images" if split == "test" else "training_images"
            for f in df.file_name:
                shutil.move(src + f"MAMI_2022_images/{folder}/{f}", f"{out_dir}/img/{img_name(f, split)}")
        shutil.rmtree(f"{raw_dir}/kagglehub")

    for split, df in splits.items():
        df["img"] = "img/" + df["file_name"].apply(lambda f: img_name(f, split))
        df["id"] = df["img"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        # label = misogynous; the four misogyny types are kept as extra columns
        df = df[["id", "img", "text", "label", "shaming", "stereotype", "objectification", "violence"]]
        check_and_save(df, out_dir, split)


for d in datasets:
    print("=" * 30, d)
    {"FHM": download_fhm, "MAMI": download_mami}[d]()
