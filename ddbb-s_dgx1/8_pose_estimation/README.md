python train_pose_resnet18.py \
  --sensor asus \
  --scene 0 \
  --gpu 0 \
  --input_dir /app/input/dataloader/ \
  --output_dir /app/output/pose/


  df -h /dev/shm
tmux ls | grep 'training_det_' | cut -d: -f1 | xargs -n1 tmux kill-session -t
