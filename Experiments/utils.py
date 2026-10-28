import os
import re
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import set_seed
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append("../Config")
from config import *


def seed_setter(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    set_seed(seed)


def read_jsonl(path, img_root):
    if not os.path.exists(path):  # pandas would treat a missing path as a json string
        raise FileNotFoundError(path)
    if os.path.getsize(path) == 0:  # e.g., no memes were augmented in a split
        return pd.DataFrame({c: pd.Series(dtype=int if c == "label" else str) for c in ["id", "img", "text", "label", "img_path"]})
    df = pd.read_json(path, lines=True, dtype={"id": str})
    df["img_path"] = img_root + df["img"]
    return df


def scale_targets(df, setting):
    # keep memes whose teacher score agrees with the binary ground truth (misaligned examples are removed)
    df = df.dropna(subset=[setting]).copy()
    df["train_label"] = df[setting].astype(int)
    agree = df["label"] == (df["train_label"] >= SCALE_THRESHOLD).astype(int)
    return df[agree].reset_index(drop=True)


# Columns of the returned splits: id, img_path, text, label (binary ground truth), train_label (training label)
#   orig            : original train/val
#   inputs          : original + counterfactual non-hateful memes in train/val
#   <prompt>_binary : original train/val, binary labels
#   <prompt>_scale  : train/val with teacher scale labels 0-9, misaligned ones removed
def load_splits(dataset, setting):
    assert setting in INPUT_SETTINGS + LABEL_SETTINGS, f"unknown setting: {setting}"
    ori, ex = ori_dir(dataset), ex_dir(dataset)
    train_df, val_df, test_df = [read_jsonl(f"{ori}{s}.jsonl", ori) for s in ["train", "val", "test"]]

    if setting == "inputs":
        train_df = pd.concat([train_df, read_jsonl(f"{ex}ex_train.jsonl", ex)], ignore_index=True)
        val_df = pd.concat([val_df, read_jsonl(f"{ex}ex_val.jsonl", ex)], ignore_index=True)

    if setting.endswith("_scale"):  # scale files point to the original images
        train_df = scale_targets(read_jsonl(f"{ex}ex_train_scale.jsonl", ori), setting)
        val_df = scale_targets(read_jsonl(f"{ex}ex_val_scale.jsonl", ori), setting)
    else:
        train_df["train_label"] = train_df["label"]
        val_df["train_label"] = val_df["label"]
    test_df["train_label"] = test_df["label"]
    return train_df, val_df, test_df


def num_classes_for(setting):
    return 10 if setting.endswith("_scale") else 2


def to_binary(preds, setting):
    # scale predictions (0-9) are mapped to TRUE (5-9) or FALSE (0-4) at test time
    if setting.endswith("_scale"):
        return [int(p >= SCALE_THRESHOLD) for p in preds]
    return [int(p) for p in preds]


# --- VLM prompts and answers ---
# orig/inputs: Yes/No prompt; Labels settings: Simple/Category prompt, answer TRUE/FALSE or a 0-9 score
def vlm_prompt(setting, text):
    if setting in INPUT_SETTINGS:
        return f"""{PROMPT_YESNO}
Caption: '{text}'
isHateful:"""
    return prompt_with_caption(PROMPTS[setting], text)


def vlm_answer(setting, target):
    if setting in INPUT_SETTINGS:
        return "Yes." if target == 1 else "No."
    if setting.endswith("_binary"):
        return "TRUE" if target == 1 else "FALSE"
    return str(target)


# returns (binary prediction, score); prediction is 2 and score -1 if the output is unclear
def parse_vlm_output(setting, text):
    text = text.strip().lower()
    if setting.endswith("_scale"):
        match = re.search(r"[0-9]", text)
        if match is None:
            return 2, -1
        score = int(match.group())
        return int(score >= SCALE_THRESHOLD), score
    yes, no = ("true", "false") if setting.endswith("_binary") else ("yes", "no")
    if yes in text:
        return 1, -1
    elif no in text:
        return 0, -1
    return 2, -1


# Resize very large images
def resize_image(image, max_size):
    width, height = image.size

    # Only resize if the image is actually larger than the max_size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def print_results(true_labels, preds):
    print("\n\n***** Printing Results *****\n\n")
    print(f"Classification Report:\n{classification_report(true_labels, preds)}", end="\n\n")

    cm = confusion_matrix(true_labels, preds, labels=[0, 1])  # unclear VLM outputs (2) are left out here
    print(f"Confusion Matrix:\n{cm}", end="\n\n")

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"False Positive Rate (FPR): {fpr:.4f}", end="\n")
    print(f"False Negative Rate (FNR): {fnr:.4f}", end="\n\n")

    print("=" * 25, end="\n\n\n\n")


# Outputs/<dataset>/<file_name>.csv with true_label, predicted_label (binary) and optional extra columns
def save_predictions(dataset, file_name, true_labels, preds, **extra):
    os.makedirs(f"Outputs/{dataset}", exist_ok=True)
    res = pd.DataFrame()
    res["true_label"] = true_labels
    res["predicted_label"] = preds
    for col, values in extra.items():
        res[col] = values
    res.to_csv(f"Outputs/{dataset}/{file_name}.csv", index=False)
