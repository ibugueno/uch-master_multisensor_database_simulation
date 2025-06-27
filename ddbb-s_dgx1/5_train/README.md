python train_seg_unet.py \
  --sensor asus \
  --gpu 0 \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-4 \
  --input_dir path/a/txts \
  --output_dir path/a/output


python train_seg_unet.py \
  --sensor asus \
  --gpu 2 \
  --input_dir /app/input/dataloader/ \
  --output_dir /app/output/seg/