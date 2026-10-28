# Label augmentation: teacher scale labels (0-9) for train and val.
# Usage: bash run_labels.sh <DATASET> <MODE>    e.g., bash run_labels.sh FHM test
#   test: 50 stratified train memes -> Outputs/<DATASET>/Sample/labels/ (check with inspect_samples.ipynb)
#   full: train + val -> Data/<DATASET>/ex/ex_{train,val}_scale.jsonl
set -e
DATASET=$1
MODE=$2

echo "Running scale labelling (teacher)..."
python3 scale_labelling.py $DATASET $MODE

echo "Parsing and filtering scale labels..."
python3 scale_filtering.py $DATASET $MODE
