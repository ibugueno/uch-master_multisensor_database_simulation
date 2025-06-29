#!/bin/bash

SCRIPT_PATH="train_depth-norm_cm_2heads_pose_resnet50.py"
INPUT_DIR="/app/input/dataloader/"
OUTPUT_DIR="/app/output/pose/"

declare -A sensors
sensors=(["asus"]=0 ["davis346"]=2 ["evk4"]=4)

for sensor in "asus" "davis346" "evk4"; do
    base_gpu="${sensors[$sensor]}"
    
    batch_size=16

    for scene in 0 1 2 3; do
        # Asignar GPU según escena
        if [ "$scene" -eq 0 ] || [ "$scene" -eq 1 ]; then
            gpu=$base_gpu
        else
            gpu=$((base_gpu + 1))
        fi

        session="training_seg_${sensor}_scene${scene}"

        tmux new-session -d -s $session "
            CUDA_VISIBLE_DEVICES=$gpu python $SCRIPT_PATH \
                --sensor $sensor \
                --scene $scene \
                --input_dir $INPUT_DIR \
                --output_dir $OUTPUT_DIR \
                --gpu 0 \
                --batch_size $batch_size ; \
            echo '[INFO] Training for $sensor scene $scene finished. Press Enter to close.' ; \
            read
        "

        echo "[INFO] Started tmux session '$session' on GPU $gpu for $sensor scene $scene with batch_size $batch_size"
    done
done
