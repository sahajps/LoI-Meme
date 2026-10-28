# Input augmentation: counterfactually non-hateful memes for train and val.
# Usage: bash run_inputs.sh <DATASET> <MODE>    e.g., bash run_inputs.sh FHM test
#   test: screens 200 hateful train memes, regenerates the first 50 NH ones -> Outputs/<DATASET>/Sample/inputs/
#   full: train + val -> Data/<DATASET>/ex/ (img/ex_<original file name>, ex_{train,val}.jsonl)
set -e
DATASET=$1
MODE=$2

echo "Running background description generator..."
python3 background_desc.py $DATASET $MODE

echo "Running hate tagging..."
python3 hate_labelling.py $DATASET $MODE

echo "Filtering data for meme augmentation..."
python3 data_for_regeneration.py $DATASET $MODE

echo "Regenerating new non-hateful captions..."
python3 non_hateful_caption_generation.py $DATASET $MODE

echo "Regenerating new meme with updated captions..."
python3 non_hateful_meme_generation.py $DATASET $MODE
