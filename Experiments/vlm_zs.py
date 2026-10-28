from gpu import select_gpus
select_gpus(1)  # the most free GPU; must run before torch is imported

from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from tqdm import tqdm
from utils import *

model_name = sys.argv[1] # enter model name, e.g., "HuggingFaceTB/SmolVLM-Instruct", etc.
dataset = sys.argv[2] # enter dataset, e.g., "FHM", "MAMI"
setting = sys.argv[3] # enter prompt setting: orig (Yes/No prompt), simple_binary, simple_scale, category_binary, category_scale
RANDOM_SEED = int(sys.argv[4]) # enter random seed for reproducibility

print("="*30)
print(f"model: {model_name}, dataset: {dataset}, setting: {setting}, seed: {RANDOM_SEED}, gpu: {os.environ['CUDA_VISIBLE_DEVICES']}")
assert setting != "inputs", "zero-shot has no training data to augment"

seed_setter(RANDOM_SEED)

# Load your data (only the test split is used)
_, _, test_df = load_splits(dataset, "orig")
print(f"test: {len(test_df)}")

# Format data for VLM
def format_data(sample):
    img_pil = Image.open(sample["img_path"]).convert("RGB")
    img_pil = resize_image(img_pil, max_size=768)

    return {
      "images": [img_pil],
      "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_pil,
                },
                {
                    "type": "text",
                    "text": vlm_prompt(setting, sample["text"]),
                }
            ],
        },
        ]
    }

test_dataset = test_df.apply(format_data, axis=1).tolist()

# --- MODEL LOADING ---
# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
    _attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_name)

# --- EVALUATION ---
def generate_text_batch(samples, max_new_tokens=5):
    text_inputs = []
    image_inputs = []

    for sample in samples:
        # Apply chat template to each sample
        text_input = processor.apply_chat_template(
            sample['messages'],
            add_generation_prompt=True
        )
        text_inputs.append(text_input)

        # Collect image for each sample
        image = sample['images']
        image_inputs.append(image)

    # Prepare the inputs for the model (batch processing)
    model_inputs = processor(
        text=text_inputs,
        images=image_inputs,
        return_tensors="pt",
        padding=True,  # Important for batch processing
        padding_side="left"
    ).to(model.device)

    # Generate text with the model
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text for all samples
    output_texts = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Return list of cleaned outputs
    return [text.strip().lower() for text in output_texts]

# Generating predictions for the test dataset
output_text = []
bs = 16
for i in tqdm(range(0, len(test_dataset), bs)):
    batch_samples = test_dataset[i:i+bs]
    batch_outputs = generate_text_batch(batch_samples, max_new_tokens=5)
    output_text.extend(batch_outputs)

# 2 (prediction) / -1 (score) if unclear
preds_labels, scores = zip(*[parse_vlm_output(setting, t) for t in output_text])
true_labels = test_df['label'].tolist()

# --- Printing Results ---
print(f"Unclear outputs: {sum(p == 2 for p in preds_labels)}")
print_results(true_labels, list(preds_labels))

# --- Saving Test Predictions ---
extra = {'predicted_score': scores} if setting.endswith("_scale") else {}
save_predictions(dataset, f"{model_name.split('/')[-1]}_ZS_{setting}", true_labels, list(preds_labels), output_text=output_text, **extra)
