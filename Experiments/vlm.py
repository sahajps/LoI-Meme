from gpu import select_gpus
select_gpus(2)  # the 2 most free GPUs for QLoRA fine-tuning; must run before torch is imported

from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm
import gc
import time
from utils import *

model_name = sys.argv[1] # enter model name, e.g., "HuggingFaceTB/SmolVLM-Instruct", etc.
dataset = sys.argv[2] # enter dataset, e.g., "FHM", "MAMI"
setting = sys.argv[3] # enter setting: orig, inputs, simple_binary, simple_scale, category_binary, category_scale
RANDOM_SEED = int(sys.argv[4]) # enter random seed for reproducibility

print("="*30)
print(f"model: {model_name}, dataset: {dataset}, setting: {setting}, seed: {RANDOM_SEED}, gpus: {os.environ['CUDA_VISIBLE_DEVICES']}")

seed_setter(RANDOM_SEED)

# Load your data
train_df, val_df, test_df = load_splits(dataset, setting)
print(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

# Format data for VLM: prompt-completion format, so the loss is computed on the answer only
# (otherwise it also covers the prompt, whose length differs a lot between the Yes/No, Simple and Category prompts)
def format_data(sample):
    img_pil = Image.open(sample["img_path"]).convert("RGB")
    img_pil = resize_image(img_pil, max_size=768)

    return {
      "images": [img_pil],
      "prompt": [
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
      ],
      "completion": [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": vlm_answer(setting, sample["train_label"]),
                }
            ],
        },
      ]
    }

train_dataset = train_df.apply(format_data, axis=1).tolist()
eval_dataset = val_df.apply(format_data, axis=1).tolist()
test_dataset = test_df.apply(format_data, axis=1).tolist()

# --- TRAINING ---
# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.bfloat16,
    quantization_config=bnb_config,
    _attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_name)

# Configure LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
    use_dora=True,
    init_lora_weights="gaussian"
)

adapter_path = f"./Chkpts/{dataset}/{model_name.split('/')[-1]}_{setting}"

# Configure training arguments using SFTConfig
training_args = SFTConfig(
    output_dir=adapter_path,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    warmup_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=25,
    save_total_limit=1,
    optim="adamw_torch_fused",
    bf16=True,
    push_to_hub=False,
    report_to="none",
    max_length=None,
    completion_only_loss=True, # loss on the answer only
    seed=RANDOM_SEED, # train is reshuffled every epoch with this seed
    data_seed=RANDOM_SEED
)

class ClearCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Clear cache every 10 steps to prevent fragmentation
        if state.global_step % 10 == 0:
            print("Clearing CUDA cache to prevent fragmentation...")
            torch.cuda.empty_cache()
            gc.collect()

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=processor,
    # MEMORY FIX: Inject the cleaner callback
    callbacks=[ClearCacheCallback()]
)

trainer.train()
trainer.save_model(training_args.output_dir)

# --- EVALUATION ---
def clear_cuda_memory():
    # Delete variables if they exist in the current global scope
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'peft_model' in globals(): del globals()['peft_model']
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

clear_cuda_memory()

# Load the fine-tuned model and processor
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.bfloat16,
    quantization_config=bnb_config,
    _attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained(model_name)

model.load_adapter(adapter_path)

def generate_text_batch(samples, max_new_tokens=5):
    text_inputs = []
    image_inputs = []

    for sample in samples:
        # Apply chat template to each sample
        text_input = processor.apply_chat_template(
            sample['prompt'],  # Use the sample without the output
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
save_predictions(dataset, f"{model_name.split('/')[-1]}_{setting}", true_labels, list(preds_labels), output_text=output_text, **extra)
