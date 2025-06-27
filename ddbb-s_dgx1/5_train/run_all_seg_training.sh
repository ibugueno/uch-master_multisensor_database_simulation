#!/bin/bash

SCRIPT_PATH="train_seg_unet.py"
INPUT_DIR="/app/input/dataloader/"
OUTPUT_DIR="/app/output/seg/"

declare -A sensors
sensors=(["asus"]=2 ["davis346"]=3 ["evk4"]=4)

for sensor in "${!sensors[@]}"; do
    gpu="${sensors[$sensor]}"
    session="training_seg_${sensor}"

    tmux new-session -d -s $session \
    "python $SCRIPT_PATH \
        --sensor $sensor \
        --input_dir $INPUT_DIR \
        --output_dir $OUTPUT_DIR \
        --gpu $gpu"
    
    echo "[INFO] Started tmux session '$session' on GPU $gpu"
done
