from sklearn.model_selection import train_test_split
from utils import *

# Teacher (gpt-5-nano) assigns a 0-9 hatefulness score to each meme (image + caption),
# once with the Simple+Scale prompt and once with the Category+Scale prompt.
dataset, mode, out_dir = get_args("labels")

for split in splits_for(mode):
    df = read_jsonl(f"{ori_dir(dataset)}{split}.jsonl")
    if mode == "test":
        df, _ = train_test_split(df, train_size=TEST_SAMPLE_SIZE, stratify=df["label"], random_state=RANDOM_SEED)
    print(f"{split}: {len(df)} memes")

    for prompt_name in ["simple_scale", "category_scale"]:
        print(f"  prompt: {prompt_name}")
        items = [
            (id, prompt_with_caption(PROMPTS[prompt_name], text), f"{ori_dir(dataset)}{img}")
            for id, text, img in zip(df.id, df.text, df.img)
        ]
        df[prompt_name + "_text"] = parallel_cached(items, OpenAI_model_response, f"{out_dir}cache/{prompt_name}_{split}.jsonl", OPENAI_WORKERS)

    save_jsonl(df, f"{out_dir}with_scale_{split}.jsonl")
