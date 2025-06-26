#!/bin/bash

# Archivo que contiene los directorios
INPUT_FILE="input/txt/folders_davis346.txt"

# Activar el entorno conda desde la ruta conocida
CONDA_PATH="/opt/conda/condabin/conda"
source /opt/conda/etc/profile.d/conda.sh

# Verificar si Conda está accesible
if ! command -v $CONDA_PATH &> /dev/null; then
    echo "Conda no está disponible en la ruta $CONDA_PATH. Verifica la instalación."
    exit 1
fi

# Activar el entorno
$CONDA_PATH activate v2e

# Escena específica a filtrar
SCENE_FILTER="scn0"

# Leer cada línea del archivo
while IFS= read -r line; do
    # Filtrar por sensor (davis346 o evk4)
    if [[ "$line" == *"davis346"* ]]; then
        height=260
        width=346
    elif [[ "$line" == *"evk4"* ]]; then
        height=720
        width=1280
    else
        echo "No se reconoció el tipo de directorio para: $line"
        continue
    fi

    # Filtrar por la escena específica
    if [[ "$line" != *"$SCENE_FILTER"* ]]; then
        echo "El directorio no pertenece a la escena filtrada ($SCENE_FILTER): $line"
        continue
    fi

    # Iterar sobre los modelos: clean y noisy
    for model in clean; do
        # Construir los directorios de salida
        output_folder="output/aedat_all_with_back_without_blur/$model/$line"

        # Ejecutar el comando
        python v2e.py -i "input/frames_all_with_back_without_blur/3_ddbs-s_with_back_without_blur/$line" \
            --overwrite \
            --input_frame_rate=1000 \
            --auto_timestamp_resolution=True \
            --dvs_exposure duration 0.005 \
            --output_folder="$output_folder" \
            --pos_thres=.15 \
            --neg_thres=.15 \
            --sigma_thres=0 \
            --output_width=$width \
            --output_height=$height \
            --disable_slomo \
            --no_preview \
            --skip_video_output \
            --dvs_aedat4 events.aedat4 \
            --dvs_params "$model"
    done

done < "$INPUT_FILE"
