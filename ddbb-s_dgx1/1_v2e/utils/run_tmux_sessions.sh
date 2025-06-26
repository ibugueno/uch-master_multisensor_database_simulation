#!/bin/bash

# Nombres de las sesiones de tmux y scripts correspondientes
SESSIONS=("extract_images_1" "extract_images_2" "extract_images_3" "extract_images_4" "extract_images_5")
SCRIPTS=("3_extract_images.py" "3_extract_images_2.py" "3_extract_images_3.py" "3_extract_images_4.py" "3_extract_images_5.py")

# Ruta al entorno virtual (ajusta según tu configuración)
CONDA_ACTIVATE="source ~/anaconda3/bin/activate v2e"

# Crear sesiones de tmux y ejecutar scripts
for i in ${!SESSIONS[@]}; do
    SESSION=${SESSIONS[$i]}
    SCRIPT=${SCRIPTS[$i]}

    # Crear y configurar cada sesión de tmux
    tmux new-session -d -s $SESSION
    tmux send-keys -t $SESSION "$CONDA_ACTIVATE" C-m  # Activa tu entorno virtual
    tmux send-keys -t $SESSION "python $SCRIPT" C-m  # Ejecuta el script
done

# Mostrar las sesiones activas
tmux list-sessions

echo "Sesiones de tmux creadas y scripts ejecutándose:"
for i in ${!SESSIONS[@]}; do
    echo "  - ${SESSIONS[$i]} ejecutando ${SCRIPTS[$i]}"
done
