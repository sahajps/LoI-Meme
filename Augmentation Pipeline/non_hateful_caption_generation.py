from utils import *

# Rewrites the hateful caption into a non-hateful, bias-reducing one that fits the background (gpt-5-nano).
dataset, mode, out_dir = get_args("inputs")


def caption_prompt(cap, desc):
    return f"""You are a helpful social media expert with deep knowledge of memes and inclusive language. Your task is to rewrite a hateful meme caption so that it becomes non-hateful and bias-reducing, while staying relevant to the background image description.

The revised caption must:
- Preserve reference to the originally targeted group or person (e.g., Muslims, Women, Immigrant, etc.), if they are mentioned. The new caption should try to retain the word related to target group in a respectful and bias-correcting way.
- Transform the message to be neutral or positively framed toward the mentioned group/person.
- Help counteract harmful stereotypes or biases.
- Remain coherent and contextually appropriate for the image.

Strict Rule: Only return the revised caption. Do not include explanations, disclaimers, or any other text.

Meme background image description: '{desc}'  

Meme caption text: '{cap}'  

Revised non-hateful meme caption:"""


for split in splits_for(mode):
    df = read_jsonl(f"{out_dir}data_for_regeneration_{split}.jsonl")
    print(f"{split}: {len(df)} memes")

    if len(df) > 0:  # an empty selection is saved as an empty file
        items = [(id, caption_prompt(cap, desc)) for id, cap, desc in zip(df.id, df.text, df.desc)]
        df["new_text"] = parallel_cached(items, OpenAI_model_response, f"{out_dir}cache/new_text_{split}.jsonl", OPENAI_WORKERS)

    save_jsonl(df, f"{out_dir}with_new_text_caption_{split}.jsonl")
