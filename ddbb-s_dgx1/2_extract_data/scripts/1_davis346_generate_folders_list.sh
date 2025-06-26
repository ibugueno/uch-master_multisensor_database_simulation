#!/bin/bash

SCRIPT_NAME=$(basename "$0")
SENSOR=$(echo "$SCRIPT_NAME" | cut -d'_' -f1)

# === Configuración de rutas absolutas ===
ROOT_INPUT_ABS="/home/ignacio.bugueno/cachefs/datasets/processed_data/moving6dpose/ddbb-s/events/${SENSOR}/events_noisy"
ROOT_OUTPUT_ABS="/home/ignacio.bugueno/cachefs/datasets/processed_data/moving6dpose/ddbb-s/events/txt"


for SCENE_ID in {0..3}; do
    SCENE_PATH="$ROOT_INPUT_ABS/scene_${SCENE_ID}/lum1000"
    OUTPUT_FILE="$ROOT_OUTPUT_ABS/folders_${SENSOR}_scn${SCENE_ID}.txt"

    # Validar existencia del directorio
    if [ ! -d "$SCENE_PATH" ]; then
        echo "[WARN] No existe: $SCENE_PATH"
        continue
    fi

    > "$OUTPUT_FILE"  # Limpiar archivo anterior

    for object_dir in "$SCENE_PATH"/*/; do
        for orientation_dir in "$object_dir"orientation_*/; do
            if [ -d "$orientation_dir" ]; then
                REL_PATH="${orientation_dir#$ROOT_INPUT_ABS/}"
                echo "$SENSOR/events_noisy/$REL_PATH" >> "$OUTPUT_FILE"
            fi
        done
    done

    echo "[OK] Generado: $OUTPUT_FILE"
done
