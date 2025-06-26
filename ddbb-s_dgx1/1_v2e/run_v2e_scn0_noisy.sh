#!/bin/bash
cd "$(dirname "$0")"

# === Configuración global ===
SCRIPT_NAME=$(basename "$0")
SCENE_FILTER=$(echo "$SCRIPT_NAME" | grep -oP 'scn[0-9]+')
SCENE_INDEX=$(echo "$SCENE_FILTER" | grep -oP '[0-9]+')
MODEL_FILTER=$(echo "$SCRIPT_NAME" | grep -oP '(noisy|clean)')
DRY_RUN=false

# === Chequear argumento opcional
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[INFO] Modo simulación activado (dry-run)"
fi

INPUT_FILE_DAVIS="input/txt/folders_davis346_${SCENE_FILTER}.txt"
INPUT_FILE_EVK4="input/txt/folders_evk4_${SCENE_FILTER}.txt"

CONDA_ENV="v2e"
CONDA_PATH="/opt/conda/condabin/conda"

# === Verificación básica
if [[ -z "$SCENE_FILTER" || -z "$MODEL_FILTER" ]]; then
    echo "Error al extraer SCENE_FILTER o MODEL_FILTER desde el nombre del script."
    exit 1
fi

echo "[INFO] Escena: $SCENE_FILTER | Modelo: $MODEL_FILTER"

# === Función para generar comandos por sensor ===
generar_comandos_v2e() {
    local input_file=$1
    local width=$2
    local height=$3
    local sensor=$4
    local gpu_id=$5
    local cmds=""
    local total=0

    while IFS= read -r line; do
        if [[ "$line" != *"$SCENE_FILTER"* ]]; then
            continue
        fi

        output_folder="output/aedat_all_with_back_without_blur/$MODEL_FILTER/$line"
        input_folder="input/frames_all_with_back_without_blur/3_ddbs-s_with_back_without_blur/$line"

        if $DRY_RUN; then
            echo "[${sensor^^}] Input:  $input_folder"
            echo "[${sensor^^}] Output: $output_folder"
        fi

        cmd="python v2e.py -i \"$input_folder\" \
            --overwrite \
            --input_frame_rate=1000 \
            --auto_timestamp_resolution=True \
            --dvs_exposure duration 0.005 \
            --output_folder=\"$output_folder\" \
            --pos_thres=.15 \
            --neg_thres=.15 \
            --sigma_thres=0 \
            --output_width=$width \
            --output_height=$height \
            --disable_slomo \
            --no_preview \
            --skip_video_output \
            --dvs_aedat4 events.aedat4 \
            --dvs_params \"$MODEL_FILTER\" \
            --use_cuda True \
            --gpu_id $gpu_id"

        cmds+="$cmd"$'\n'
        ((total++))
    done < "$input_file"

    if $DRY_RUN; then
        echo "[${sensor^^}] Total comandos generados: $total"
    fi

    echo "$cmds"
}

# === Generar comandos por sensor ===
CMDS_DAVIS=$(generar_comandos_v2e "$INPUT_FILE_DAVIS" 346 260 "davis346" "$SCENE_INDEX")
CMDS_EVK4=$(generar_comandos_v2e "$INPUT_FILE_EVK4" 1280 720 "evk4" "$((SCENE_INDEX + 1))")

# === Modo dry-run termina aquí ===
if $DRY_RUN; then
    echo "[INFO] Fin de simulación (dry-run). No se ejecutó ningún tmux ni procesamiento."
    exit 0
fi

# === Lanzar sesiones tmux ===
tmux new-session -d -s "v2e_davis346_${SCENE_FILTER}_${MODEL_FILTER}" bash -c "
source /opt/conda/etc/profile.d/conda.sh && conda activate $CONDA_ENV
$CMDS_DAVIS
exec bash
"

tmux new-session -d -s "v2e_evk4_${SCENE_FILTER}_${MODEL_FILTER}" bash -c "
source /opt/conda/etc/profile.d/conda.sh && conda activate $CONDA_ENV
$CMDS_EVK4
exec bash
"

# === Mensaje final ===
echo "==> Sesiones tmux lanzadas:"
tmux ls | grep v2e_

echo "==> Puedes usar:"
echo "tmux attach -t v2e_davis346_${SCENE_FILTER}_${MODEL_FILTER}"
echo "tmux attach -t v2e_evk4_${SCENE_FILTER}_${MODEL_FILTER}"
