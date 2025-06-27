python train_det_fasterrcnn.py \
  --sensor asus \
  --gpu 0 \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-4 \
  --input_dir path/a/txts \
  --output_dir path/a/output


python train_det_fasterrcnn.py \
  --sensor asus \
  --scene 0 \
  --gpu 0 \
  --input_dir /app/input/dataloader/ \
  --output_dir /app/output/det/


  df -h /dev/shm
