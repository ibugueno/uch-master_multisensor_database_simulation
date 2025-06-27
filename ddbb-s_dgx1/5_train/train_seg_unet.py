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

    orientation_objects = defaultdict(list)

    # Construye el dict con indices por objeto
    for i, path in enumerate(paths):
        if "orientation_88_-6_-34" in str(path):
            obj = str(path).split("/")[-3]
            orientation_objects[obj].append((path, i))

    # Procesa cada objeto
    for obj, path_idx_list in orientation_objects.items():
        if not path_idx_list:
            continue

        # Ordena por nombre de archivo
        path_idx_list.sort(key=lambda x: os.path.basename(x[0]))

        # Selecciona el índice de la mitad
        selected_path, idx = path_idx_list[len(path_idx_list)//2]

        # Carga imagenes
        img = Image.open(selected_path).convert("RGB")
        pred_img = Image.fromarray((preds[idx].squeeze().cpu().numpy() > 0.5).astype(np.uint8)*255).convert("RGB")
        target_img = Image.fromarray((targets[idx].squeeze().cpu().numpy()).astype(np.uint8)*255).convert("RGB")

        # Anota
        img = annotate(img, f"{obj} Input")
        pred_img = annotate(pred_img, "Predicted")
        target_img = annotate(target_img, "Expected")

        # Concatena
        concatenated = Image.new("RGB", (img.width + pred_img.width + target_img.width, img.height))
        concatenated.paste(img, (0,0))
        concatenated.paste(pred_img, (img.width,0))
        concatenated.paste(target_img, (img.width + pred_img.width,0))

        # Guarda
        concatenated.save(out_dir / f"example_{obj}_orientation_88_-6_-34.png")

    print("[INFO] Saved examples for orientation_88_-6_-34 per object")



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    with open(metrics_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "MeanIoU_Object", "MeanIoU_Background", "MeanIoU"])

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
            iou_objs, iou_bgs = [], []
            per_object = defaultdict(lambda: {"iou_obj": [], "iou_bg": []})

            with torch.no_grad():
                for imgs, masks, paths in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                    imgs, masks = imgs.to(device), masks.to(device)
                    outputs = model(imgs)
                    preds = torch.sigmoid(outputs) > 0.5

                    iou_obj, iou_bg = compute_iou(preds, masks)
                    iou_objs.extend(iou_obj.cpu().numpy())
                    iou_bgs.extend(iou_bg.cpu().numpy())

                    for i, path in enumerate(paths):
                        obj = str(path).split("/")[-3]
                        per_object[obj]["iou_obj"].append(iou_obj[i].item())
                        per_object[obj]["iou_bg"].append(iou_bg[i].item())

            mean_iou_obj = np.mean(iou_objs)
            mean_iou_bg = np.mean(iou_bgs)
            mean_iou = (mean_iou_obj + mean_iou_bg) / 2

            # Escribir en CSV
            writer.writerow([epoch+1, mean_iou_obj, mean_iou_bg, mean_iou])

            # Guardar métricas en .txt
            with open(out_path / "metrics.txt", "w") as f:
                f.write(f"Epoch {epoch+1}\n")
                f.write(f"Mean IoU Object: {mean_iou_obj:.4f}\n")
                f.write(f"Mean IoU Background: {mean_iou_bg:.4f}\n")
                f.write(f"Mean IoU: {mean_iou:.4f}\n\n")
                for obj in per_object:
                    f.write(f"Object: {obj}\n")
                    f.write(f" IoU Object: {np.mean(per_object[obj]['iou_obj']):.4f}\n")
                    f.write(f" IoU Background: {np.mean(per_object[obj]['iou_bg']):.4f}\n\n")

            # Guardar mejor modelo y ejemplos
            if mean_iou > best_mean_iou:
                best_mean_iou = mean_iou
                torch.save(model.state_dict(), out_path / "best_model.pth")
                save_example_outputs(preds, masks, paths, out_path)

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
