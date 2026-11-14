---
pretty_name: Labels or Input? Augmentations of FHM and MAMI
language:
- en
task_categories:
- image-classification
- text-classification
tags:
- hateful-memes
- misogyny
- multimodal
- data-augmentation
- counterfactual
size_categories:
- 10K<n<100K
configs:
- config_name: FHM_inputs
  data_files:
  - split: train
    path: FHM/ex/ex_train.jsonl
  - split: validation
    path: FHM/ex/ex_val.jsonl
- config_name: FHM_labels
  data_files:
  - split: train
    path: FHM/ex/ex_train_scale.jsonl
  - split: validation
    path: FHM/ex/ex_val_scale.jsonl
- config_name: MAMI_inputs
  data_files:
  - split: train
    path: MAMI/ex/ex_train.jsonl
  - split: validation
    path: MAMI/ex/ex_val.jsonl
- config_name: MAMI_labels
  data_files:
  - split: train
    path: MAMI/ex/ex_train_scale.jsonl
  - split: validation
    path: MAMI/ex/ex_val_scale.jsonl
---

# Labels or Input? Augmentations of FHM and MAMI

> **Content warning:** this dataset contains hateful, misogynistic and offensive text and images.

This dataset holds two kinds of augmentation for the train and validation splits of the
[Facebook Hateful Memes (FHM)](https://huggingface.co/datasets/willgrobots/hateful_memes_zipped) and
[MAMI](https://www.kaggle.com/datasets/nrizwan/mami-dataset) (SemEval-2022 Task 5) datasets. The test splits are not augmented.

- **Inputs**: new non-hateful counterfactual memes. They are made from hateful memes whose caption is hateful but whose image is not. gpt-5-nano describes the image and rewrites the caption into a non-hateful one, and gemini-3.1-flash-lite-image generates a new meme from the description and the new caption. Every new meme has label 0.
- **Labels**: a 0-9 hatefulness score from a teacher model (gpt-5-nano) for every original meme. Each meme gets two scores, one from a Simple prompt and one from a Category prompt.

The original memes are **not** included here. Download FHM and MAMI from the links above.

## Structure

```
FHM/ex/  and  MAMI/ex/
├── img/ex_<original file name>   # new memes, in the same format as the original image (FHM: .png, MAMI: .jpg)
├── ex_train.jsonl                # Inputs
├── ex_val.jsonl
├── ex_train_scale.jsonl          # Labels
└── ex_val_scale.jsonl
```

| File | FHM | MAMI | Fields |
|---|---|---|---|
| `ex_train.jsonl` | 1,275 | 820 | `id`, `img`, `text`, `label` |
| `ex_val.jsonl` | 40 | 98 | `id`, `img`, `text`, `label` |
| `ex_train_scale.jsonl` | 8,500 | 9,000 | `id`, `img`, `text`, `label`, `simple_scale`, `simple_scale_text`, `category_scale`, `category_scale_text` |
| `ex_val_scale.jsonl` | 640 | 1,000 | same as above |

- `id`: the id of the original meme, kept as a string because FHM ids have leading zeros (e.g., `"08291"`). In `ex_*.jsonl` it is the meme the new one was made from.
- `img`: in `ex_*.jsonl`, the new image, relative to `ex/`. In `ex_*_scale.jsonl`, the original image as `img/<original file name>`; for MAMI, this includes the split suffix described below.
- `text`: the new non-hateful caption in `ex_*.jsonl`, and the original caption in `ex_*_scale.jsonl`.
- `label`: the original binary label: hateful for FHM, misogynous for MAMI. It is always 0 for the new memes.
- `simple_scale` / `category_scale`: the teacher's 0-9 score; 5 or more means hateful.
- `simple_scale_text` / `category_scale_text`: the teacher's raw answer.

The scale files keep every meme. For training on the scores, keep only memes whose score agrees with the binary label (0-4 for label 0, 5-9 for label 1).

## Splits of the original datasets

- FHM: train = `train` (8,500); validation = `dev_seen` ∪ `dev_unseen` (640).
- MAMI: train = `train.tsv` (9,000); validation = `validation.tsv` (1,000). Five file names appear in both the training and test image folders, so they carry a split suffix: `_tr` for train, `_val` for validation and `_t` for test (e.g., `15001_val.jpg`).

## Usage

```python
from datasets import load_dataset
inputs = load_dataset("sahajps/LoI-Meme", "FHM_inputs")   # captions and image paths; also FHM_labels, MAMI_inputs, MAMI_labels

# images: download everything, e.g., next to the original datasets in Data/<DATASET>/ori/
from huggingface_hub import snapshot_download
snapshot_download(repo_id="sahajps/LoI-Meme", repo_type="dataset", local_dir="Data")
```

The original memes and captions remain under the terms of the FHM and MAMI datasets.

## 📚 Cite Us

If you find this work helpful and use our dataset or methodology, please cite:

```bibtex
@article{singh2025labels,
  title={Labels or Input? Rethinking Augmentation in Multimodal Hate Detection},
  author={Singh, Sahajpreet and Jaidka, Kokil and Mukerjee, Subhayan},
  journal={arXiv preprint arXiv:2508.11808},
  year={2025}
}
```
