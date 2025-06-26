#!/bin/bash

docker rm -fv ignacio_moving6dpose-s_v2e

docker run -it --gpus '"device=0,1,2,3,4,5,6,7"' --name ignacio_moving6dpose-s_v2e -v /home/ignacio.bugueno/cachefs/datasets/processed_data/moving6dpose/ddbb-s/frames:/app/input -v /home/ignacio.bugueno/cachefs/datasets/processed_data/moving6dpose/ddbb-s/events:/app/output ignacio_moving6dpose-s_v2e
