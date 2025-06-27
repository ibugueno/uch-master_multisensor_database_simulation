#!/bin/bash

SCRIPT_PATH="train_seg_unet.py"
INPUT_DIR="/app/input/dataloader/"
OUTPUT_DIR="/app/output/seg/"

declare -A sensors
sensors=(["asus"]=2 ["davis346"]=3 ["evk4"]=4)

for sensor in "${!sensors[@]}"; do
    gpu="${sensors[$sensor]}"
    
    # Batch size dinámico
    if [ "$sensor" == "evk4" ]; then
        batch_size=16
    else
        batch_size=16
    fi

    for scene in 0 1 2 3; do
        session="training_seg_${sensor}_scene${scene}"

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
