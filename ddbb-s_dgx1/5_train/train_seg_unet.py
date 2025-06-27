# file: train_binary_unet_segmentation.py

import os
import argparse
import yaml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import segmentation_models_pytorch as smp
import csv
import random
from collections import defaultdict

SENSOR_SHAPES = {
    "asus": (360, 640),
    "davis346": (260, 346),
    "evk4": (720, 1280)
}

VAL_ORIENTATIONS = [
    "orientation_39_17_-102",
    "orientation_19_31_21",
    "orientation_-125_66_-116",
    "orientation_88_-6_-34"
]

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

        # Resize explícito a la mitad solo si evk4
        if "evk4" in self.image_paths[idx]:
            resized_h, resized_w = SENSOR_SHAPES["evk4"][0] // 2, SENSOR_SHAPES["evk4"][1] // 2
            img = img.resize((resized_w, resized_h), Image.BILINEAR)
            mask = mask.resize((resized_w, resized_h), Image.NEAREST)  # usar NEAREST para máscaras binarias

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        mask = transforms.ToTensor()(mask)
        mask = (mask > 0.5).float()

        return img, mask, self.image_paths[idx]


def get_transform(sensor):
    h, w = SENSOR_SHAPES[sensor]
    crop_size = (h//2, w//2) if sensor == "evk4" else (h, w)

    return transforms.Compose([
        transforms.RandomResizedCrop(size=crop_size, scale=(0.8, 1.0)),
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

def get_datasets(sensor, input_dir, scene):
    data_txt = os.path.join(input_dir, f"{sensor}_data_scene_{scene}.txt")
    mask_txt = os.path.join(input_dir, f"{sensor}_mask-seg_scene_{scene}.txt")

    img_paths, mask_paths = load_paths(data_txt, mask_txt)

    train_data, train_mask, val_data, val_mask = [], [], [], []

    for img, mask in zip(img_paths, mask_paths):
        if any(orient in img for orient in VAL_ORIENTATIONS):
            val_data.append(img)
            val_mask.append(mask)
        else:
            train_data.append(img)
            train_mask.append(mask)

    transform_train = get_transform(sensor)
    train_dataset = SegmentationDataset(train_data, train_mask, transform=transform_train)
    val_dataset = SegmentationDataset(val_data, val_mask)
    return train_dataset, val_dataset

def compute_iou(preds, targets):
    preds = preds > 0.5
    targets = targets > 0.5

    # Object (1)
    intersection_obj = (preds & targets).float().sum(dim=[1,2,3])
    union_obj = (preds | targets).float().sum(dim=[1,2,3])
    iou_obj = (intersection_obj + 1e-6) / (union_obj + 1e-6)

    # Background (0)
    preds_bg = ~preds
    targets_bg = ~targets
    intersection_bg = (preds_bg & targets_bg).float().sum(dim=[1,2,3])
    union_bg = (preds_bg | targets_bg).float().sum(dim=[1,2,3])
    iou_bg = (intersection_bg + 1e-6) / (union_bg + 1e-6)

    return iou_obj, iou_bg

def annotate(image, label):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = ImageFont.load_default()

    # Usa textbbox si existe, fallback a estimación simple
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        # Estimación manual si no existe textbbox ni textsize
        text_width = len(label) * font.size * 0.6
        text_height = font.size

    bg_rect = [0, 0, int(text_width + 10), int(text_height + 10)]
    draw.rectangle(bg_rect, fill="white")
    draw.text((5, 5), label, fill="black", font=font)
    return image

def save_example_outputs(preds, targets, paths, out_path):
    out_dir = out_path / "examples"
    out_dir.mkdir(exist_ok=True)

    # 🔍 DEBUG: verificar todos los paths que contienen orientation_88_-6_-34
    #print("\n[DEBUG] Checking all paths with orientation_88_-6_-34 (grouped by object and sorted by filename)")
    orientation_objects = defaultdict(list)
    for i, path in enumerate(paths):
        if "orientation_88_-6_-34" in str(path):
            obj = str(path).split("/")[-3]
            orientation_objects[obj].append((i, path))

    for obj in sorted(orientation_objects.keys()):
        sorted_entries = sorted(orientation_objects[obj], key=lambda x: os.path.basename(str(x[1])))
        #for i, path in sorted_entries:
            #print(f"[DEBUG] i={i}, obj={obj}, path={path}")
    #print("[DEBUG] Finished listing paths\n")

    orientation_objects = defaultdict(list)
    for i, path in enumerate(paths):
        if "orientation_88_-6_-34" in str(path):
            obj = str(path).split("/")[-3]
            orientation_objects[obj].append(i)

    for obj, indices in orientation_objects.items():
        if not indices:
            continue
        # ordenar indices por nombre de archivo
        sorted_indices = sorted(indices, key=lambda x: os.path.basename(str(paths[x])))

        if "scene_2" in str(out_path):
            idx = sorted_indices[int(0.6 * len(sorted_indices))]
        else:
            idx = sorted_indices[len(sorted_indices)//2]  # mitad para otras escenas


        img = Image.open(paths[idx]).convert("RGB")
        pred_img = Image.fromarray((preds[idx].squeeze().cpu().numpy() > 0.5).astype(np.uint8)*255).convert("RGB")
        target_img = Image.fromarray((targets[idx].squeeze().cpu().numpy()).astype(np.uint8)*255).convert("RGB")

        # Calcular altura total y ancho máximo
        total_height = img.height + pred_img.height + target_img.height
        max_width = max(img.width, pred_img.width, target_img.width)

        concatenated = Image.new("RGB", (max_width, total_height))
        concatenated.paste(img, (0,0))
        concatenated.paste(pred_img, (0, img.height))
        concatenated.paste(target_img, (0, img.height + pred_img.height))


        concatenated.save(out_dir / f"example_{obj}_orientation_88_-6_-34.png")



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_metrics(preds, targets):
    preds_bin = preds > 0.5
    targets_bin = targets > 0.5

    TP = (preds_bin & targets_bin).float().sum(dim=[1,2,3])
    FP = (preds_bin & ~targets_bin).float().sum(dim=[1,2,3])
    FN = (~preds_bin & targets_bin).float().sum(dim=[1,2,3])

    # === IoU ===
    union = TP + FP + FN
    iou = TP / union
    iou[union == 0] = float('nan')

    # === Precision ===
    denom_p = TP + FP
    precision = TP / denom_p
    precision[denom_p == 0] = float('nan')

    # === Recall ===
    denom_r = TP + FN
    recall = TP / denom_r
    recall[denom_r == 0] = float('nan')

    # === Dice ===
    dice = (2 * TP) / (2 * TP + FP + FN)
    dice[(2 * TP + FP + FN) == 0] = float('nan')

    return iou, dice, precision, recall, 


def train_model(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_set, val_set = get_datasets(args.sensor, args.input_dir, args.scene)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    out_path = Path(args.output_dir) / f"{args.sensor}_scene_{args.scene}"
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "config.yaml", 'w') as f:
        yaml.dump(vars(args), f)

    best_mean_iou = 0
    metrics_csv = out_path / "metrics.csv"

    # === Apertura inicial del CSV para encabezados ===
    with open(metrics_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "MeanIoU", "MeanDice", "MeanPrecision", "MeanRecall"])

    for epoch in range(args.epochs):
        model.train()
        for imgs, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        model.eval()
        iou_list, dice_list, prec_list, recall_list = [], [], [], []
        per_object = defaultdict(lambda: {"iou": [], "dice": [], "precision": [], "recall": []})

        # === NUEVO: acumuladores globales ===
        all_preds = []
        all_masks = []
        all_paths = []

        with torch.no_grad():
            for imgs, masks, paths in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                preds = torch.sigmoid(outputs) > 0.5

                iou, dice, precision, recall = compute_metrics(preds, masks)

                iou_list.extend(iou.cpu().numpy())
                dice_list.extend(dice.cpu().numpy())
                prec_list.extend(precision.cpu().numpy())
                recall_list.extend(recall.cpu().numpy())

                for i, path in enumerate(paths):
                    obj = str(path).split("/")[-3]
                    per_object[obj]["iou"].append(iou[i].item())
                    per_object[obj]["dice"].append(dice[i].item())
                    per_object[obj]["precision"].append(precision[i].item())
                    per_object[obj]["recall"].append(recall[i].item())

                all_preds.append(preds.cpu())
                all_masks.append(masks.cpu())
                all_paths.extend(paths)

        # === Promedios globales (ignorando NaN) ===
        mean_iou = np.nanmean(iou_list) if len(iou_list) > 0 else 0.0
        mean_dice = np.nanmean(dice_list) if len(dice_list) > 0 else 0.0
        mean_precision = np.nanmean(prec_list) if len(prec_list) > 0 else 0.0
        mean_recall = np.nanmean(recall_list) if len(recall_list) > 0 else 0.0

        # === Escritura en CSV ===
        with open(metrics_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, mean_iou, mean_dice, mean_precision, mean_recall])

        # === Guardar métricas en .txt ===
        with open(out_path / "metrics.txt", "w") as f:
            f.write(f"Epoch {epoch+1}\n")
            f.write(f"Mean IoU: {mean_iou:.4f}\n")
            f.write(f"Mean Dice: {mean_dice:.4f}\n")
            f.write(f"Mean Precision: {mean_precision:.4f}\n")
            f.write(f"Mean Recall: {mean_recall:.4f}\n\n")
            for obj in per_object:
                f.write(f"Object: {obj}\n")
                f.write(f" IoU: {np.nanmean(per_object[obj]['iou']):.4f}\n")
                f.write(f" Dice: {np.nanmean(per_object[obj]['dice']):.4f}\n")
                f.write(f" Precision: {np.nanmean(per_object[obj]['precision']):.4f}\n")
                f.write(f" Recall: {np.nanmean(per_object[obj]['recall']):.4f}\n\n")

        # === Guardar mejor modelo y ejemplos ===
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            torch.save(model.state_dict(), out_path / "best_model.pth")

            all_preds_cat = torch.cat(all_preds, dim=0)
            all_masks_cat = torch.cat(all_masks, dim=0)
            save_example_outputs(all_preds_cat, all_masks_cat, all_paths, out_path)

    print(f"[DONE] Best model saved at {out_path / 'best_model.pth'}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor', required=True, choices=["asus", "davis346", "evk4"])
    parser.add_argument('--scene', required=True, type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"[INFO] Training for sensor: {args.sensor}, scene: {args.scene}")
    train_model(args)
