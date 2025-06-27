#!/bin/bash

SCRIPT_PATH="train_det_fasterrcnn.py"
INPUT_DIR="/app/input/dataloader/"
OUTPUT_DIR="/app/output/det/"

sensor="evk4"
batch_size=16

# GPUs fijos para cada scene
declare -a gpus=(2 3 6 7)

for scene in 0 1 2 3; do
    gpu=${gpus[$scene]}
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
