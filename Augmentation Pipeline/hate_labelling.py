from utils import *

# Hatefulness attribution (gpt-5-nano): is the caption hateful? is the background (description) hateful?
dataset, mode, out_dir = get_args("inputs")


def prompt_t(txt):
    return "You are a social media content moderator. You are given a text caption of a meme and your task is to identify whether it's hateful? Answer in 'Yes' or 'No' only. \n\n>>> Image Caption: " + txt + " \n\n>>> isHateful:"


def prompt_d(des):
    return "You are a social media content moderator. You are given a text description of the background image of a meme and your task is to identify whether it's hateful? Answer in 'Yes' or 'No' only. \n\n>>> Image Description: " + des + " \n\n>>> isHateful:"


for split in splits_for(mode):
    df = read_jsonl(f"{out_dir}with_desc_{split}.jsonl")
    print(f"{split}: {len(df)} memes")

    print("  caption")
    items = [(id, prompt_t(txt)) for id, txt in zip(df.id, df.text)]
    df["hateLabelText"] = parallel_cached(items, OpenAI_model_response, f"{out_dir}cache/hatelabel_text_{split}.jsonl", OPENAI_WORKERS)

    print("  background description")
    items = [(id, prompt_d(des)) for id, des in zip(df.id, df.desc)]
    df["hateLabelDesc"] = parallel_cached(items, OpenAI_model_response, f"{out_dir}cache/hatelabel_desc_{split}.jsonl", OPENAI_WORKERS)

    save_jsonl(df, f"{out_dir}with_hatelabels_with_desc_{split}.jsonl")
