from utils import *

# Regenerates the meme from the background description and the new caption (text-only, Gemini).
# Images are saved as img/ex_<original file name> in the original's format, annotations as ex_<split>.jsonl
# full: -> Data/<DATASET>/ex/
# test: -> Outputs/<DATASET>/Sample/inputs/
dataset, mode, out_dir = get_args("inputs")
save_dir = out_dir if mode == "test" else ex_dir(dataset)
os.makedirs(save_dir + "img", exist_ok=True)


def extra_inverted_comma_remover(text):
    text = text.strip()
    if text[:1] in ['"', "'"]:
        text = text[1:]
    if text[-1:] in ['"', "'"]:
        text = text[:-1]
    return text


def generate(id, txt, desc, img):
    prompt = f"You are a creative social media expert with deep knowledge of internet memes. Your task is to generate a humorous and engaging meme image that is non-hateful. Use the given background description and overlay the provided text in a visually appealing way. \n\nBackground image description: '{desc}' \n\nMeme text: '{txt}' \n\nEnsure the meme aligns with popular internet humor while remaining appropriate and widely relatable."
    return Gemini_image_response(prompt, save_dir + img, f"{out_dir}logs/log_{id}.txt")


for split in splits_for(mode):
    df = read_jsonl(f"{out_dir}with_new_text_caption_{split}.jsonl")
    if len(df) == 0:
        print(f"{split}: no memes to regenerate")
        open(f"{save_dir}ex_{split}.jsonl", "w").close()
        continue
    df = df.dropna(subset=["new_text"])
    print(f"{split}: {len(df)} memes")

    df["orig_img"], df["orig_text"] = df.img, df.text
    df["img"] = "img/ex_" + df.img.apply(os.path.basename)
    df["text"] = df.new_text.apply(extra_inverted_comma_remover)
    df["label"] = 0  # as we are generating non-hateful memes, we set all labels to 0

    with ThreadPoolExecutor(max_workers=GEMINI_WORKERS) as executor:
        list(tqdm(executor.map(generate, df.id, df.text, df.desc, df.img), total=len(df)))

    df = df[[os.path.exists(save_dir + i) for i in df.img]]
    print(f"{split}: {len(df)} non-hateful memes generated")

    # id is the original meme's id; test mode keeps the original meme and description for inspection
    cols = ["id", "img", "text", "label"] + (["orig_img", "orig_text", "desc"] if mode == "test" else [])
    save_jsonl(df[cols], f"{save_dir}ex_{split}.jsonl")
    print(f"Saved to {save_dir}ex_{split}.jsonl")
