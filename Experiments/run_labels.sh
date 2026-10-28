# #!/bin/bash
# Labels experiments: Simple/Category prompt x binary/scaled labels, the prompt is fed to the student.
# Fine-tuned text models and VLMs (same training as the Inputs experiments), and zero-shot VLMs with the same prompts.
# GPUs are picked automatically: the 2 most free for VLM fine-tuning, the most free one otherwise (see gpu.py)
# Usage: bash run_labels.sh [model groups] [datasets]
#   e.g., nohup bash run_labels.sh > Logs/labels.out &
#         nohup bash run_labels.sh "vlm zs" MAMI > Logs/labels_vlm_MAMI.out &
# model groups: text (ALBERT, BERT, RoBERTa), vlm (QLoRA fine-tuning), zs (zero-shot VLMs)
MODEL_GROUPS=" ${1:-text vlm zs} "
DATASETS=${2:-"FHM MAMI"}

TEXT_MODELS="albert/albert-base-v2 google-bert/bert-base-uncased FacebookAI/roberta-base"
VLMS="HuggingFaceTB/SmolVLM-Instruct Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2-VL-7B-Instruct"
SETTINGS="simple_binary simple_scale category_binary category_scale"

for DATASET in $DATASETS; do
  if [[ $MODEL_GROUPS == *" text "* ]]; then
    for MODEL in $TEXT_MODELS; do
      for SETTING in $SETTINGS; do
        echo "$MODEL $DATASET $SETTING 42"
        python3 text_only.py "$MODEL" $DATASET $SETTING 42
      done
    done
  fi

  if [[ $MODEL_GROUPS == *" vlm "* ]]; then
    for MODEL in $VLMS; do
      for SETTING in $SETTINGS; do
        echo "$MODEL $DATASET $SETTING 42"
        python3 vlm.py "$MODEL" $DATASET $SETTING 42
      done
    done
  fi

  if [[ $MODEL_GROUPS == *" zs "* ]]; then
    for MODEL in $VLMS; do
      for SETTING in $SETTINGS; do
        echo "$MODEL $DATASET $SETTING (zero-shot) 42"
        python3 vlm_zs.py "$MODEL" $DATASET $SETTING 42
      done
    done
  fi
done
