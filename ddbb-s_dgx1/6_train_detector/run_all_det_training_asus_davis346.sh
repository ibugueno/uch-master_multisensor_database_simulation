#!/bin/bash

SCRIPT_PATH="train_det_fasterrcnn.py"
INPUT_DIR="/app/input/dataloader/"
OUTPUT_DIR="/app/output/det/"

declare -A base_gpus
base_gpus=(["asus"]=0 ["davis346"]=4)

for sensor in "asus" "davis346"; do
    base_gpu=${base_gpus[$sensor]}
    
    # Batch size fijo en 8 para ambos
    batch_size=8

    for scene in 0 1 2 3; do
        gpu=$((base_gpu + scene))
        session="training_det_${sensor}_scene${scene}"

        tmux new-session -d -s $session "
            python $SCRIPT_PATH \
                --sensor $sensor \
                --scene $scene \
                --input_dir $INPUT_DIR \
                --output_dir $OUTPUT_DIR \
                --gpu $gpu \
                --batch_size $batch_size ; \
            echo '[INFO] Training for $sensor scene $scene finished. Press Enter to close.' ; \
            read
        "

        echo "[INFO] Started tmux session '$session' on GPU $gpu for scene $scene with batch_size $batch_size"
    done
done
