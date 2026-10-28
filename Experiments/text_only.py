from gpu import select_gpus
select_gpus(1)  # the most free GPU; must run before torch is imported

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from utils import *

model_name = sys.argv[1] # enter model name, e.g., "bert-base-uncased", "roberta-base", etc.
dataset = sys.argv[2] # enter dataset, e.g., "FHM", "MAMI"
setting = sys.argv[3] # enter setting: orig, inputs, simple_binary, simple_scale, category_binary, category_scale
RANDOM_SEED = int(sys.argv[4]) # enter random seed for reproducibility
device = "cuda"

print("="*30)
print(f"model: {model_name}, dataset: {dataset}, setting: {setting}, seed: {RANDOM_SEED}, gpu: {os.environ['CUDA_VISIBLE_DEVICES']}")

seed_setter(RANDOM_SEED)

# Load your data
train_df, val_df, test_df = load_splits(dataset, setting)
print(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

# orig/inputs: raw caption only; Labels settings: Simple/Category prompt + caption as a sentence pair
prompt = PROMPTS.get(setting)
num_classes = num_classes_for(setting)
max_len = 64 if prompt is None else 512

tokenizer = AutoTokenizer.from_pretrained(model_name)
if prompt is not None:
    # Category prompts (~700 tokens) don't fit in 512: the longer sequence (the prompt) is cut first, from the left,
    # so the output instruction at its end is kept; captions are only cut if longer than ~256 tokens
    tokenizer.truncation_side = "left"

# --- Dataset ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        if prompt is None:
            self.encodings = tokenizer(list(texts), truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        else:
            captions = [f"Caption: '{t}'" for t in texts]
            self.encodings = tokenizer([prompt] * len(captions), captions, truncation="longest_first", padding="max_length", max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels.values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

train_dataset = TextDataset(train_df['text'], train_df['train_label'], tokenizer, max_len)
val_dataset = TextDataset(val_df['text'], val_df['train_label'], tokenizer, max_len)
test_dataset = TextDataset(test_df['text'], test_df['train_label'], tokenizer, max_len)

# train is reshuffled every epoch (seeded); val/test keep their order so predictions align with the data
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(RANDOM_SEED))
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# --- Custom Model ---
class CustomBERTClassifier(nn.Module):
    def __init__(self):
        super(CustomBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        # Freeze all layers except the last two encoder layers
        for param in self.bert.parameters():
            param.requires_grad = False
        if hasattr(self.bert.encoder, "layer"): # BERT, RoBERTa
            last_layers = self.bert.encoder.layer[-2:]
        else: # ALBERT shares one layer across depth, so its (only) layer group is trained
            last_layers = self.bert.encoder.albert_layer_groups[-1:]
        for param in last_layers.parameters():
            param.requires_grad = True

    def forward(self, **inputs):
        outputs = self.bert(**inputs)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(cls_output)
        return logits

def predict(loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            logits = model(**inputs)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    return preds

model = CustomBERTClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

# --- Training Loop with Early Stopping ---
best_f1 = 0
patience = 3
counter = 0
num_epochs = 100
chkpt_path = f"Chkpts/{dataset}/{model_name.split('/')[-1]}_{setting}.pt"
os.makedirs(f"Chkpts/{dataset}", exist_ok=True)

# --- Training ---
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}", end=" | ")
    model.train()
    total_loss = 0

    for batch in train_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)

        logits = model(**inputs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Train Loss: {total_loss / len(train_loader):.4f}", end=" | ")

    # --- Validation (on binary labels, also for scale settings) ---
    preds = to_binary(predict(val_loader), setting)
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
scores = predict(test_loader)
preds = to_binary(scores, setting)
true_labels = test_df['label'].tolist()

# --- Printing Results ---
print_results(true_labels, preds)

# --- Saving Test Predictions ---
extra = {'predicted_score': scores} if setting.endswith("_scale") else {}
save_predictions(dataset, f"{model_name.split('/')[-1]}_{setting}", true_labels, preds, **extra)
