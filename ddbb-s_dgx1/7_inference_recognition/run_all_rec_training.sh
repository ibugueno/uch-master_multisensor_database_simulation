#!/bin/bash

SCRIPT_PATH="eval_fasterrcnn_detection_metrics.py"
INPUT_DIR="/app/input/dataloader/"
OUTPUT_DIR="/app/output/recognition/"
MODEL_DIR="/app/output/det/"

batch_size=16

declare -A sensor_gpu
sensor_gpu=(["asus"]=0 ["davis346"]=2 ["evk4"]=4)

for sensor in "asus" "davis346" "evk4"; do
    base_gpu=${sensor_gpu[$sensor]}

    for scene in 0 1 2 3; do
        gpu=$((base_gpu + scene / 2))
        session=\"eval_rec_${sensor}_scene${scene}\"
        model_path=\"${MODEL_DIR}${sensor}_scene_${scene}/fasterrcnn_model.pth\"

        tmux new-session -d -s $session \"
            python $SCRIPT_PATH \\\\\\
                --sensor $sensor \\\\\\
                --scene $scene \\\\\\
                --model_path $model_path \\\\\\
                --input_dir $INPUT_DIR \\\\\\
                --output_dir $OUTPUT_DIR \\\\\\
                --gpu $gpu \\\\\\
                --batch_size $batch_size ; \\\\\\
            echo '[INFO] Evaluation for $sensor scene $scene finished. Press Enter to close.' ; \\\\\\
            read
        \"

        echo \"[INFO] Started tmux session '$session' on GPU $gpu for scene $scene with batch_size $batch_size\"
    done
done

# === Usage ===
# chmod +x run_eval_fasterrcnn.sh
# ./run_eval_fasterrcnn.sh
