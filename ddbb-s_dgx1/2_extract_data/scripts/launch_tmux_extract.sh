#!/bin/bash

SCRIPT_DIR=$(pwd)
TXT_DIR="../input/events/txt"

for sensor in davis346 evk4; do
  for scn in 0 1 2 3; do
    session_name="${sensor}_scn${scn}"

    if [ "$sensor" = "davis346" ]; then
      script="3_davis346_noisy_extract_data.py"
    else
      script="4_evk4_noisy_extract_data.py"
    fi

    txt_file="${TXT_DIR}/folders_${sensor}_scn${scn}.txt"
    cmd="python $script --txt_file $txt_file; bash"

    tmux new-session -d -s "$session_name" "cd $SCRIPT_DIR && $cmd"
    echo "Sesión creada: $session_name → ejecutando: $cmd"
  done
done

echo -e "\n======================="
echo "Accede a cada sesión con:"
for sensor in davis346 evk4; do
  for scn in 0 1 2 3; do
    session_name="${sensor}_scn${scn}"
    echo "tmux attach -t $session_name"
  done
done
echo "=======================\n"
