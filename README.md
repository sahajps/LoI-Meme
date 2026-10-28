# Labels or Input 🤔
Code and data for **Labels or Input? Rethinking Augmentation in Multimodal Hate Detection**. We compare two kinds of augmentation on two datasets (FHM and MAMI):
- 🏷️ **Labels**: a teacher (gpt-5-nano) adds 0-9 hatefulness scores to train and val, with Simple or Category prompts.
- 🖼️ **Inputs**: counterfactually non-hateful memes are generated for train and val with an LLM–VLM pipeline.

The test sets are never augmented.

## 📦 Dataset
Download and standardize both datasets (details and folder structure in `Data/README.md`):
```bash
cd Data
python3 download_data.py
```

## 🚀 Run
Ensure you're using `python 3.11.11` and install dependencies:
```bash
python3.11 -m venv LoI_env
source LoI_env/bin/activate
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1 --no-build-isolation
```
Put your API keys in `Config/openai_key.txt` (one key), `Config/gemini_keys.txt` (one key per line) and `Config/hf_token.txt` (HF read token, needed for a fast FHM download). Model names, prompts and other shared settings are in `Config/config.py`.

### 🧬 Augmentation Pipeline
Each pipeline has a `test` mode that runs on a small sample, so outputs can be checked with `inspect_samples.ipynb` before the `full` run:
```bash
cd Augmentation\ Pipeline
bash run_labels.sh FHM test   # 50 memes -> Outputs/FHM/Sample/labels/
bash run_inputs.sh FHM test   # 50 regenerated memes -> Outputs/FHM/Sample/inputs/

bash run_labels.sh FHM full   # -> Data/FHM/ex/ex_{train,val}_scale.jsonl
bash run_inputs.sh FHM full   # -> Data/FHM/ex/ (img/ex_*, ex_{train,val}.jsonl)
```
| Step | Model |
|---|---|
| Scale labels (teacher) | gpt-5-nano |
| Background description | gpt-5-nano |
| Hate attribution (caption / background) | gpt-5-nano |
| Non-hateful caption | gpt-5-nano |
| Meme regeneration (text-only, from description + caption) | gemini-3.1-flash-lite-image |

Intermediate files go to `Outputs/<DATASET>/{Sample,Full}/{labels,inputs}/`. API calls are cached in their `cache/` subfolder, so an interrupted run resumes where it stopped.

### 📊 Benchmarking
| | Inputs experiments | Labels experiments |
|---|---|---|
| Run script | `run_inputs.sh` | `run_labels.sh` |
| Settings | `orig`, `inputs` (original vs. original + counterfactual memes) | `simple_binary`, `simple_scale`, `category_binary`, `category_scale` |
| Input to the model | caption (text models), Yes/No prompt + caption (VLMs) | Simple/Category prompt + caption |
| Fine-tuned models | ALBERT, BERT, RoBERTa, ViT, DeiT, BEiT, CLIP, SmolVLM, Qwen2.5-VL-3B, Qwen2-VL-7B | ALBERT, BERT, RoBERTa, SmolVLM, Qwen2.5-VL-3B, Qwen2-VL-7B |
| Zero-shot VLMs | Yes/No prompt | Simple/Category prompts |

Both parts use the same training: encoders fine-tune their last two layers (AdamW, lr 5e-5, batch 128, early stopping on validation weighted-F1), and VLMs use QLoRA (4-bit, DoRA r=8, α=8, dropout 0.1, 2 epochs, lr 5e-5, effective batch 32, loss on the answer only). `*_scale` settings train on teacher scores (0-9) that are mapped to binary (5-9 = hateful) at test time.
```bash
cd Experiments
bash run_inputs.sh                 # all Inputs runs
bash run_labels.sh "vlm zs" MAMI   # model groups (text, image, clip, vlm, zs) and datasets can be selected
```
GPUs are picked automatically when a run starts: the 2 with the most free memory for VLM fine-tuning, otherwise the one with the most free memory. To choose yourself, set `CUDA_VISIBLE_DEVICES`, e.g., `CUDA_VISIBLE_DEVICES=5 bash run_labels.sh text`.
Each run saves its test predictions to `Outputs/<DATASET>/<model>_<setting>.csv` (`<model>_ZS_<setting>.csv` for zero-shot). To compute evaluation metrics and result tables, use `Experiments/analysis.ipynb`.

### 🔍 Data Analysis
- `label_dist_analysis.ipynb`: label counts per split and setting, teacher agreement, meme typology.
- `fhm_overlap_analysis.ipynb`: memes shared between the original FHM splits.
- `mami_overlap_analysis.ipynb`: file names and memes shared between the MAMI splits.
