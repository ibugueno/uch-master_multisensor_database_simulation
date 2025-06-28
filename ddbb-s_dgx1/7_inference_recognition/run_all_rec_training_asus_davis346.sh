#!/bin/bash

SCRIPT_PATH="eval_fasterrcnn_detection_metrics.py"
INPUT_DIR="/app/input/dataloader/"
OUTPUT_DIR="/app/output/recognition/"
MODEL_DIR="/app/output/det/"

declare -A base_gpus
base_gpus=(["asus"]=0 ["davis346"]=4)

for sensor in "asus" "davis346"; do
    base_gpu=${base_gpus[$sensor]}
    batch_size=16

    for scene in 0 1 2 3; do
        gpu=$((base_gpu + scene))
        session="eval_rec_${sensor}_scene${scene}"
        model_path="${MODEL_DIR}${sensor}_scene_${scene}/fasterrcnn_model.pth"

        tmux new-session -d -s $session "
            python $SCRIPT_PATH \\
                --sensor $sensor \\
                --scene $scene \\
                --model_path $model_path \\
                --input_dir $INPUT_DIR \\
                --output_dir $OUTPUT_DIR \\
                --gpu $gpu \\
                --batch_size $batch_size ; \\
            echo '[INFO] Evaluation for $sensor scene $scene finished. Press Enter to close.' ; \\
            read
        "

        echo "[INFO] Started tmux session '$session' on GPU $gpu for scene $scene with batch_size $batch_size"
    done
done

# === Usage ===
# chmod +x run_eval_fasterrcnn.sh
# ./run_eval_fasterrcnn.sh

# The MODEL_DIR has been updated to point to /app/output/det/ as per your environment structure.
