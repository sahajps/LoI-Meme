from utils import *

# Background description of each hateful meme with gpt-5-nano (excluding the overlaid text).
# Only hateful memes can be regenerated, so non-hateful memes are skipped.
dataset, mode, out_dir = get_args("inputs")

query = "Analyze and describe the given image's visuals in short. Do not include the description about the text written on the image."

for split in splits_for(mode):
    df = read_jsonl(f"{ori_dir(dataset)}{split}.jsonl")
    df = df[df.label == 1]
    if mode == "test":
        df = df.sample(min(TEST_POOL_SIZE, len(df)), random_state=RANDOM_SEED)
    print(f"{split}: {len(df)} hateful memes")

    items = [(id, query, f"{ori_dir(dataset)}{img}") for id, img in zip(df.id, df.img)]
    df["desc"] = parallel_cached(items, OpenAI_model_response, f"{out_dir}cache/desc_{split}.jsonl", OPENAI_WORKERS)

    # memes without a description (failed call or refusal) can't be checked for hateful backgrounds
    missing = df.desc.fillna("").str.strip() == ""
    if missing.any():
        print(f"  skipping {missing.sum()} memes without a description")
    save_jsonl(df[~missing], f"{out_dir}with_desc_{split}.jsonl")
