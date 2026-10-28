from gpu import select_gpus
select_gpus(1)  # the most free GPU; must run before torch is imported

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from utils import *

model_name = sys.argv[1] # enter model name, e.g., "openai/clip-vit-base-patch32", etc.
dataset = sys.argv[2] # enter dataset, e.g., "FHM", "MAMI"
setting = sys.argv[3] # enter setting: orig, inputs (CLIP is not used for the Labels part)
RANDOM_SEED = int(sys.argv[4]) # enter random seed for reproducibility
device = "cuda"

print("="*30)
print(f"model: {model_name}, dataset: {dataset}, setting: {setting}, seed: {RANDOM_SEED}, gpu: {os.environ['CUDA_VISIBLE_DEVICES']}")
assert setting in INPUT_SETTINGS, f"{setting} is not used for CLIP"

seed_setter(RANDOM_SEED)

# Load your data
train_df, val_df, test_df = load_splits(dataset, setting)
print(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

processor = AutoProcessor.from_pretrained(model_name)

# --- Dataset ---
class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['img_path']).convert("RGB")
        text = row['text']
        label = torch.tensor(row['train_label'], dtype=torch.long)

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": label
        }

train_dataset = MultimodalDataset(train_df, processor)
val_dataset = MultimodalDataset(val_df, processor)
test_dataset = MultimodalDataset(test_df, processor)

# train is reshuffled every epoch (seeded); val/test keep their order so predictions align with the data
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(RANDOM_SEED))
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# --- Custom Model ---
class CustomCLIPClassifier(nn.Module):
    def __init__(self):
        super(CustomCLIPClassifier, self).__init__()
        self.clip = AutoModel.from_pretrained(model_name)

        # Freeze all layers except the last two layers of both the vision and text encoders
        for param in self.clip.parameters():
            param.requires_grad = False

        for layer in list(self.clip.vision_model.encoder.layers[-2:]) + list(self.clip.text_model.encoder.layers[-2:]):
            for param in layer.parameters():
                param.requires_grad = True

        hidden_size = self.clip.config.projection_dim
        self.classifier = nn.Linear(hidden_size * 2, 2)

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_loss=False
        )
        image_embed = outputs.image_embeds
        text_embed = outputs.text_embeds
        fused = torch.cat([image_embed, text_embed], dim=1)
        logits = self.classifier(fused)
        return logits

def predict(loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(pixel_values, input_ids, attention_mask)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    return [int(p) for p in preds]

model = CustomCLIPClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

# --- Training Loop with Early Stopping ---
best_f1 = 0
patience = 3
counter = 0
num_epochs = 100
chkpt_path = f"Chkpts/{dataset}/{model_name.split('/')[-1]}_{setting}.pt"
os.makedirs(f"Chkpts/{dataset}", exist_ok=True)

# Training
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}", end=" | ")
    model.train()
    total_loss = 0

    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(pixel_values, input_ids, attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Train Loss: {total_loss / len(train_loader):.4f}", end=" | ")

    # --- Validation ---
    preds = predict(val_loader)
    val_acc = accuracy_score(val_df['label'], preds)
    val_f1 = f1_score(val_df['label'], preds, average='weighted')
    print(f"Val Accuracy: {val_acc:.4f} | weighted-F1: {val_f1:.4f}")

    # --- Early Stopping ---
    if val_f1 > best_f1:
        best_f1 = val_f1
        counter = 0
        torch.save(model.state_dict(), chkpt_path)
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# --- Final Evaluation ---
model.load_state_dict(torch.load(chkpt_path))
preds = predict(test_loader)
true_labels = test_df['label'].tolist()

# --- Printing Results ---
print_results(true_labels, preds)

# --- Saving Test Predictions ---
save_predictions(dataset, f"{model_name.split('/')[-1]}_{setting}", true_labels, preds)
