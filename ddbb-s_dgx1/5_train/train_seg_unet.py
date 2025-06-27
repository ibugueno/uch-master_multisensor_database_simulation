# file: train_binary_unet_segmentation.py

import os
import argparse
import yaml
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from tqdm import tqdm
import random
import csv
from PIL import ImageDraw, ImageFont
import time

class_mapping = {
    'almohada': 0, 'arbol': 1, 'avion': 2, 'boomerang': 3,
    'caja_amarilla': 4, 'caja_azul': 5, 'carro_rojo': 6, 'clorox': 7,
    'dino': 8, 'jarron': 9, 'lysoform': 10, 'mobil': 11,
    'paleta': 12, 'pelota': 13, 'sombrero': 14, 'tarro': 15, 'zapatilla': 16
}

EXAMPLE_OBJECT_PATH = "orientation_88_-6_-34"

SENSOR_SHAPES = {
    "asus": (360, 640),
    "davis346": (260, 346),
    "evk4": (720, 1280)
}

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask) > 0
        return img, mask.float(), self.image_paths[idx]

def get_transform(sensor):
    h, w = SENSOR_SHAPES[sensor]
    return transforms.Compose([
        transforms.RandomResizedCrop(size=(h, w), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor()
    ])

def load_paths(data_txt, mask_txt):
    with open(data_txt) as f:
        image_paths = [line.strip() for line in f]
    with open(mask_txt) as f:
        mask_paths = [line.strip() for line in f]
    return image_paths, mask_paths

def get_datasets(sensor, input_dir):
    train_data, train_mask = load_paths(
        os.path.join(input_dir, f"{sensor}_data_scene_0.txt"),
        os.path.join(input_dir, f"{sensor}_mask-seg_scene_0.txt")
    )
    val_data, val_mask = [], []
    for i in [1, 2, 3]:
        d, m = load_paths(
            os.path.join(input_dir, f"{sensor}_data_scene_{i}.txt"),
            os.path.join(input_dir, f"{sensor}_mask-seg_scene_{i}.txt")
        )
        val_data.extend(d)
        val_mask.extend(m)
    transform_train = get_transform(sensor)
    return (
        SegmentationDataset(train_data, train_mask, transform=transform_train),
        SegmentationDataset(val_data, val_mask)
    )

def calculate_metrics(preds, targets):
    preds = preds > 0.5
    targets = targets > 0.5
    cm = confusion_matrix(targets.view(-1).cpu(), preds.view(-1).cpu(), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return {"IoU": iou, "Precision": precision, "Recall": recall, "F1-Score": f1}

def annotate(image, label):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = ImageFont.load_default()
    text_size = draw.textsize(label, font=font)
    bg_rect = [0, 0, text_size[0] + 10, text_size[1] + 10]
    draw.rectangle(bg_rect, fill="white")
    draw.text((5, 5), label, fill="black", font=font)
    return image

def save_example_outputs(preds, targets, paths, out_path):
    from collections import defaultdict

    out_dir = out_path / "examples"
    out_dir.mkdir(exist_ok=True)

    # Agrupar por orientación
    orientation_groups = defaultdict(list)
    for i, p in enumerate(paths):
        if EXAMPLE_OBJECT_PATH in str(p):
            orientation_groups[EXAMPLE_OBJECT_PATH].append((i, p))

    for orientation, samples in orientation_groups.items():
        # Ordenar por nombre de archivo
        samples_sorted = sorted(samples, key=lambda x: str(x[1]))
        mid_index = len(samples_sorted) // 2
        idx, p = samples_sorted[mid_index]

        pred = preds[idx]
        target = targets[idx]

        img = Image.open(p).convert("RGB")
        pred_img = Image.fromarray((pred.squeeze().numpy() > 0.5).astype(np.uint8) * 255).convert("RGB")
        target_img = Image.fromarray((target.squeeze().numpy()).astype(np.uint8) * 255).convert("RGB")

        img = annotate(img, "Input")
        pred_img = annotate(pred_img, "Predicted")
        target_img = annotate(target_img, "Expected")

        concatenated = Image.new("RGB", (img.width + pred_img.width + target_img.width, img.height))
        concatenated.paste(img, (0, 0))
        concatenated.paste(pred_img, (img.width, 0))
        concatenated.paste(target_img, (img.width + pred_img.width, 0))
        concatenated.save(out_dir / f"example_{orientation}.png")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_model(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_set, val_set = get_datasets(args.sensor, args.input_dir)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    num_classes = len(class_mapping)
    model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    out_path = Path(args.output_dir) / args.sensor
    out_path.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    with open(out_path / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    best_iou = 0
    metrics_csv = out_path / "metrics.csv"

    with open(metrics_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["epoch", "Mean IoU", "Pixel Acc"] + [f"Class_{cls}_{metric}" for cls in range(num_classes) for metric in ["IoU", "Precision", "Recall", "F1"]]
        writer.writerow(header)

        for epoch in range(args.epochs):
            model.train()
            for imgs, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
                imgs, masks = imgs.to(device), masks.to(device).long().squeeze(1)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            model.eval()
            total_tp = np.zeros(num_classes)
            total_fp = np.zeros(num_classes)
            total_fn = np.zeros(num_classes)
            correct_pixels = 0
            total_pixels = 0

            with torch.no_grad():
                for imgs, masks, paths in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                    imgs, masks = imgs.to(device), masks.to(device).long().squeeze(1)
                    outputs = model(imgs)
                    preds = torch.argmax(outputs, dim=1)

                    preds_np = preds.cpu().numpy()
                    masks_np = masks.cpu().numpy()

                    correct_pixels += (preds_np == masks_np).sum()
                    total_pixels += np.prod(masks_np.shape)

                    for cls in range(num_classes):
                        tp = ((preds_np == cls) & (masks_np == cls)).sum()
                        fp = ((preds_np == cls) & (masks_np != cls)).sum()
                        fn = ((preds_np != cls) & (masks_np == cls)).sum()
                        total_tp[cls] += tp
                        total_fp[cls] += fp
                        total_fn[cls] += fn

            precision = total_tp / (total_tp + total_fp + 1e-6)
            recall = total_tp / (total_tp + total_fn + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)

            pixel_acc = correct_pixels / total_pixels
            mean_iou = np.mean(iou)

            # Save metrics
            row = [epoch + 1, mean_iou, pixel_acc]
            for cls in range(num_classes):
                row += [iou[cls], precision[cls], recall[cls], f1[cls]]
            writer.writerow(row)

            # Save txt summary
            with open(out_path / "metrics.txt", "w") as f:
                f.write(f"Epoch {epoch+1}\n")
                f.write(f"Pixel Acc: {pixel_acc:.4f}\n")
                f.write(f"Mean IoU: {mean_iou:.4f}\n")
                for cls in range(num_classes):
                    f.write(f"Class {cls}: IoU={iou[cls]:.4f}, Precision={precision[cls]:.4f}, Recall={recall[cls]:.4f}, F1={f1[cls]:.4f}\n")

            # Save best model
            if mean_iou > best_iou:
                best_iou = mean_iou
                torch.save(model.state_dict(), out_path / "model.pth")

    print(f"[DONE] Best model saved to {out_path / 'model.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor', required=True, choices=["asus", "davis346", "evk4"], help='Sensor name')
    parser.add_argument('--input_dir', required=True, help='Directory containing .txt lists')
    parser.add_argument('--output_dir', required=True, help='Directory to store output models and results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    print(f"[INFO] Training for sensor: {args.sensor}")
    train_model(args)