from utils import *

# Keeps hateful memes whose caption is hateful but background is not (NH), the target of regeneration.
dataset, mode, out_dir = get_args("inputs")


def label_map(x):
    if x is None or pd.isna(x):
        return None
    x = x.lower()
    if "yes" in x:
        return 1
    elif "no" in x:
        return 0
    # output in some other format
    return None


for split in splits_for(mode):
    df = read_jsonl(f"{out_dir}with_hatelabels_with_desc_{split}.jsonl")

    df.hateLabelText = df.hateLabelText.apply(label_map)
    df.hateLabelDesc = df.hateLabelDesc.apply(label_map)

    # meme typology of hateful memes: H/N for (background, caption)
    print(f"\n##### {split}: {len(df)} hateful memes #####")
    print(f"Unparsable: caption {df.hateLabelText.isna().sum()}, background {df.hateLabelDesc.isna().sum()}")
    for typ, (d, t) in {"HH": (1, 1), "HN": (1, 0), "NH": (0, 1), "NN": (0, 0)}.items():
        print(f"{typ}: {((df.hateLabelDesc == d) & (df.hateLabelText == t)).sum()}")

    # taking the case of harmful caption and harmless description
    df = df[(df.label == 1) & (df.hateLabelText == 1) & (df.hateLabelDesc == 0)]
    if mode == "test":
        df = df.head(TEST_SAMPLE_SIZE)
    print(f"Selected for regeneration: {len(df)}")

    df = df[["id", "img", "label", "text", "desc"]]
    save_jsonl(df, f"{out_dir}data_for_regeneration_{split}.jsonl")
