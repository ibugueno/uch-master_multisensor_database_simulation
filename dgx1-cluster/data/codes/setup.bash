#!/bin/bash

# Elimina libffi de conda y reinstala dependencias del sistema
conda remove --force libffi
apt-get update
apt-get install --reinstall -y libffi7 libffi-dev libllvm12
ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 /usr/lib/x86_64-linux-gnu/libffi.so
ln -sf /usr/lib/x86_64-linux-gnu/libLLVM-12.so.1 /usr/lib/x86_64-linux-gnu/libLLVM-12.so
ldconfig

# Reinstala libffi y scipy en conda
apt-get update
apt-get install --reinstall -y libffi7 libffi-dev
conda install -c conda-forge libffi --force-reinstall -y
conda install -c conda-forge scipy --force-reinstall -y
ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 /usr/lib/x86_64-linux-gnu/libffi.so
ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 /usr/lib/x86_64-linux-gnu/libffi.so.8
ldconfig

# Configura variables de entorno y reinicia Xvfb
export DISPLAY=:99
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
pkill Xvfb
rm /tmp/.X99-lock
Xvfb :99 -screen 0 1024x768x24 &
