### Data Setup

Run `python3 download_data.py` from this folder. It downloads both datasets and writes them in the same format, so every experiment script works on either dataset unchanged.

**Dataset sources:**
- FHM: [https://huggingface.co/datasets/willgrobots/hateful_memes_zipped](https://huggingface.co/datasets/willgrobots/hateful_memes_zipped) (put an HF read token in `Config/hf_token.txt`)
- MAMI: [https://www.kaggle.com/datasets/nrizwan/mami-dataset](https://www.kaggle.com/datasets/nrizwan/mami-dataset) (SemEval-2022 Task 5)

**Folder structure:**
```
Data/<DATASET>/                    # DATASET = FHM | MAMI
├── ori/                           # Original dataset
│   ├── img/*                      # Meme images
│   ├── raw/                       # Untouched original annotation files
│   ├── train.jsonl                # id, img, text, label  (img is relative to ori/)
│   ├── val.jsonl
│   └── test.jsonl                 # Never augmented
└── ex/                            # Our augmentation of train and val
    ├── img/ex_<original file name> # Counterfactual non-hateful memes (same format as the original)
    ├── ex_train.jsonl             # Inputs: id (of the original meme), img (relative to ex/), text (new caption), label (0)
    ├── ex_val.jsonl
    ├── ex_train_scale.jsonl       # Labels: original memes (img relative to ori/) + simple_scale, simple_scale_text, category_scale, category_scale_text
    └── ex_val_scale.jsonl
```
`ex/` is filled by `Augmentation Pipeline/run_inputs.sh <DATASET> full` and `run_labels.sh <DATASET> full`.

**Split notes:**
- FHM: `train` = train (8,500); `val` = dev_seen ∪ dev_unseen (640; the two share 400 memes); `test` = test_seen + test_unseen (3,000). See `fhm_overlap_analysis.ipynb` for the overlaps between the original splits.
- MAMI: `train` (9,000) and `val` (1,000) are the dataset's `train.tsv` and `validation.tsv`, which together are the official 10,000 training memes; `test` = the official test set (1,000). `label` is `misogynous`; the columns `shaming`, `stereotype`, `objectification` and `violence` are kept. All images are in one `img/` folder with their original names, except the 5 names that exist in both the training and test folders, which get a split suffix (e.g., `15001_val.jpg`, `15001_t.jpg`). See `mami_overlap_analysis.ipynb`.
- `ex_*_scale.jsonl` keep every meme with both teacher scores (0-9). The experiments use only memes whose score agrees with the binary label (0-4 for label 0, 5-9 for label 1).

**Hugging Face release:** the `ex/` folders of both datasets hold everything we add, so they can go into one HF repo (`FHM/ex/`, `MAMI/ex/`).
