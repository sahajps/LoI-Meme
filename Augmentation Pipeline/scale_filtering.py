import re
from utils import *

# Parses the teacher's 0-9 scores and reports how often they agree with the binary ground truth.
# All memes are saved with both scores; the experiments keep only the agreeing ones
# (label 0 & score 0-4, or label 1 & score 5-9).
dataset, mode, out_dir = get_args("labels")


def parse_scale(x):
    if x is None or pd.isna(x):
        return None
    x = str(x).strip()
    if re.fullmatch(r"[0-9]", x):
        return int(x)
    match = re.search(r"score\D{0,20}([0-9])\b", x, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.findall(r"\b([0-9])\b", x)
    return int(match[-1]) if match else None


for split in splits_for(mode):
    df = read_jsonl(f"{out_dir}with_scale_{split}.jsonl")
    print(f"\n##### {split}: {len(df)} memes #####")

    for prompt_name in ["simple_scale", "category_scale"]:
        df[prompt_name] = df[prompt_name + "_text"].apply(parse_scale).astype("Int64")
        scale = df[prompt_name]
        agree = (df["label"] == (scale >= SCALE_THRESHOLD).astype("Int64")) & scale.notna()

        print(f"\n[{prompt_name}]")
        print(f"Unparsable responses: {scale.isna().sum()}")
        print(f"Score distribution: { {int(k): int(v) for k, v in scale.value_counts().sort_index().items()} }")
        print(f"Agreement with binary label: {agree.sum()}/{len(df)} ({100 * agree.mean():.2f}%)")
        print(f"Kept per label: {df[agree].label.value_counts().to_dict()}")

    # img is the original meme's image (relative to Data/<DATASET>/ori/)
    df = df[["id", "img", "text", "label", "simple_scale", "simple_scale_text", "category_scale", "category_scale_text"]]
    save_dir = out_dir if mode == "test" else ex_dir(dataset)
    os.makedirs(save_dir, exist_ok=True)
    save_jsonl(df, f"{save_dir}ex_{split}_scale.jsonl")
    print(f"Saved to {save_dir}ex_{split}_scale.jsonl")
