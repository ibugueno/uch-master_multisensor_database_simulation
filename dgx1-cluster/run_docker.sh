#!/bin/bash

docker rm -fv ignacio_thesis_blender_simulation_2

nvidia-docker run -it -e NVIDIA_VISIBLE_DEVICES=0 --name ignacio_thesis_blender_simulation_2 -v /home/ignacio.bugueno/cachefs/docker/test_blender/data/input:/data/input -v /home/ignacio.bugueno/cachefs/docker/test_blender/data/output:/data/output -v /home/ignacio.bugueno/cachefs/docker/test_blender/data/codes:/data/codes ignacio_thesis_blender_simulation_2

sleep 10; docker logs ignacio_thesis_blender_simulation_2

