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

def dice_score(preds, targets):
    preds = preds > 0.5
    targets = targets > 0.5
    intersection = (preds & targets).float().sum((1, 2))
    union = preds.float().sum((1, 2)) + targets.float().sum((1, 2))
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice.mean().item()

def weighted_pixel_acc(preds, targets):
    correct = (preds == targets).sum().item()
    total = preds.numel()
    return correct / total

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
    out_dir = out_path / "examples"
    print(f"save_example_outputs: {out_dir}")
    out_dir.mkdir(exist_ok=True)
    for i, path in enumerate(paths):
        if any(orient in str(path) for orient in VAL_ORIENTATIONS):
            img = Image.open(path).convert("RGB")
            pred_img = Image.fromarray((preds[i].squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255).convert("RGB")
            target_img = Image.fromarray((targets[i].squeeze().cpu().numpy()).astype(np.uint8) * 255).convert("RGB")
            img = annotate(img, "Input")
            pred_img = annotate(pred_img, "Predicted")
            target_img = annotate(target_img, "Expected")
            concatenated = Image.new("RGB", (img.width + pred_img.width + target_img.width, img.height))
            concatenated.paste(img, (0, 0))
            concatenated.paste(pred_img, (img.width, 0))
            concatenated.paste(target_img, (img.width + pred_img.width, 0))
            orientation_name = [o for o in VAL_ORIENTATIONS if o in str(path)][0]
            concatenated.save(out_dir / f"example_{orientation_name}.png")

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

    config = vars(args)
    with open(out_path / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    best_dice = 0
    metrics_csv = out_path / "metrics.csv"

    with open(metrics_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "PixelAcc", "MeanIoU", "Dice", "WeightedAcc"])

        for epoch in range(args.epochs):
            model.train()
            for imgs, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)  # sin squeeze
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            model.eval()
            dices, ious, accs, weights = [], [], [], []
            per_object = defaultdict(lambda: {"dice": [], "iou": [], "acc": []})

            with torch.no_grad():
                for imgs, masks, paths in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                    imgs, masks = imgs.to(device), masks.to(device)

                    outputs = model(imgs)  # sin squeeze
                    preds = torch.sigmoid(outputs) > 0.5

                    dices.append(dice_score(preds, masks))
                    ious.append(((preds.bool() & masks.bool()).sum().item()) / ((preds.bool() | masks.bool()).sum().item() + 1e-6))
                    accs.append(weighted_pixel_acc(preds, masks))
                    for i, path in enumerate(paths):
                        obj = str(path).split("/")[-3]
                        per_object[obj]["dice"].append(dice_score(preds[i:i+1], masks[i:i+1]))
                        per_object[obj]["iou"].append(((preds[i].bool() & masks[i].bool()).sum().item()) / ((preds[i].bool() | masks[i].bool()).sum().item() + 1e-6))
                        per_object[obj]["acc"].append(weighted_pixel_acc(preds[i], masks[i]))

                save_example_outputs(preds, masks, paths, out_path)

            mean_dice = np.mean(dices)
            mean_iou = np.mean(ious)
            mean_acc = np.mean(accs)

            writer.writerow([epoch+1, mean_acc, mean_iou, mean_dice, mean_acc])

            with open(out_path / "metrics.txt", "w") as f:
                f.write(f"Epoch {epoch+1}\n")
                f.write(f"PixelAcc: {mean_acc:.4f}\n")
                f.write(f"MeanIoU: {mean_iou:.4f}\n")
                f.write(f"Dice: {mean_dice:.4f}\n\n")
                for obj in per_object:
                    f.write(f"Object: {obj}\n")
                    f.write(f" Dice: {np.mean(per_object[obj]['dice']):.4f}\n")
                    f.write(f" IoU: {np.mean(per_object[obj]['iou']):.4f}\n")
                    f.write(f" Acc: {np.mean(per_object[obj]['acc']):.4f}\n\n")

            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), out_path / "best_model.pth")

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
