#!/bin/bash

docker rm -fv ignacio_moving6dpose-s_extract_data

docker run -it --gpus '"device=0"' --name ignacio_moving6dpose-s_extract_data -v /home/ignacio.bugueno/cachefs/datasets/processed_data/moving6dpose/ddbb-s:/app/input ignacio_moving6dpose-s_extract_data
