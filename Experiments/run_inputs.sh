# #!/bin/bash
# Inputs experiments: original vs original + counterfactual memes (settings orig, inputs), and zero-shot VLMs (Yes/No prompt)
# GPUs are picked automatically: the 2 most free for VLM fine-tuning, the most free one otherwise (see gpu.py)
# Usage: bash run_inputs.sh [model groups] [datasets]
#   e.g., nohup bash run_inputs.sh > Logs/inputs.out &
#         nohup bash run_inputs.sh "vlm zs" FHM > Logs/inputs_vlm_FHM.out &
# model groups: text (ALBERT, BERT, RoBERTa), image (ViT, DeiT, BEiT), clip, vlm (QLoRA fine-tuning), zs (zero-shot VLMs)
MODEL_GROUPS=" ${1:-text image clip vlm zs} "
DATASETS=${2:-"FHM MAMI"}

TEXT_MODELS="albert/albert-base-v2 google-bert/bert-base-uncased FacebookAI/roberta-base"
IMAGE_MODELS="google/vit-base-patch16-224 facebook/deit-base-patch16-224 microsoft/beit-base-patch16-224-pt22k-ft22k"
CLIP_MODELS="openai/clip-vit-base-patch32"
VLMS="HuggingFaceTB/SmolVLM-Instruct Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2-VL-7B-Instruct"

for DATASET in $DATASETS; do
  if [[ $MODEL_GROUPS == *" text "* ]]; then
    for MODEL in $TEXT_MODELS; do
      for SETTING in orig inputs; do
        echo "$MODEL $DATASET $SETTING 42"
        python3 text_only.py "$MODEL" $DATASET $SETTING 42
      done
    done
  fi

  if [[ $MODEL_GROUPS == *" image "* ]]; then
    for MODEL in $IMAGE_MODELS; do
      for SETTING in orig inputs; do
        echo "$MODEL $DATASET $SETTING 42"
        python3 image_only.py "$MODEL" $DATASET $SETTING 42
      done
    done
  fi

  if [[ $MODEL_GROUPS == *" clip "* ]]; then
    for MODEL in $CLIP_MODELS; do
      for SETTING in orig inputs; do
        echo "$MODEL $DATASET $SETTING 42"
        python3 clip.py "$MODEL" $DATASET $SETTING 42
      done
    done
  fi

  if [[ $MODEL_GROUPS == *" vlm "* ]]; then
    for MODEL in $VLMS; do
      for SETTING in orig inputs; do
        echo "$MODEL $DATASET $SETTING 42"
        python3 vlm.py "$MODEL" $DATASET $SETTING 42
      done
    done
  fi

  if [[ $MODEL_GROUPS == *" zs "* ]]; then
    for MODEL in $VLMS; do
      echo "$MODEL $DATASET orig (zero-shot) 42"
      python3 vlm_zs.py "$MODEL" $DATASET orig 42
    done
  fi
done
