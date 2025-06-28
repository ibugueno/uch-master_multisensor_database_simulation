"""
train_pose_estimation.py (final full script with parse_args and clean main)
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import csv
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
from pathlib import Path
import random


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

CLASS_MAPPING = {
    'almohada': 1,
    'arbol': 2,
    'avion': 3,
    'boomerang': 4,
    'caja_amarilla': 5,
    'caja_azul': 6,
    'carro_rojo': 7,
    'clorox': 8,
    'dino': 9,
    'jarron': 10,
    'lysoform': 11,
    'mobil': 12,
    'paleta': 13,
    'pelota': 14,
    'sombrero': 15,
    'tarro': 16,
    'zapatilla': 17
}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_paths(data_txt, label_txt):
    with open(data_txt) as f:
        image_paths = [line.strip() for line in f]
    with open(label_txt) as f:
        label_paths = [line.strip() for line in f]
    return image_paths, label_paths

def get_datasets(sensor, input_dir, scene):
    data_txt = os.path.join(input_dir, f"{sensor}_data_scene_{scene}.txt")
    label_txt = os.path.join(input_dir, f"{sensor}_pose6d-abs-10ms_scene_{scene}.txt")
    img_paths, label_paths = load_paths(data_txt, label_txt)

    train_imgs, train_lbls, val_imgs, val_lbls = [], [], [], []
    for img, lbl in zip(img_paths, label_paths):
        if any(orient in img for orient in VAL_ORIENTATIONS):
            val_imgs.append(img)
            val_lbls.append(lbl)
        else:
            train_imgs.append(img)
            train_lbls.append(lbl)

    train_dataset = PoseDataset(train_imgs, train_lbls, sensor)
    val_dataset = PoseDataset(val_imgs, val_lbls, sensor)
    return train_dataset, val_dataset


class PoseDataset(Dataset):
    def __init__(self, image_paths, label_paths, sensor):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.sensor = sensor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # fixed input size for ResNet18
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        if 'evk4' in img_path:
            resized_h, resized_w = SENSOR_SHAPES['evk4'][0] // 2, SENSOR_SHAPES['evk4'][1] // 2
            img = img.resize((resized_w, resized_h), Image.BILINEAR)
            scale_x = resized_w / orig_w
            scale_y = resized_h / orig_h
        else:
            scale_x = scale_y = 1.0

        label_path = self.label_paths[idx]
        with open(label_path) as f:
            lines = f.readlines()[1]
            data = [float(x) for x in lines.strip().split(',')]
            xmin, ymin, xmax, ymax = map(float, data[:4])
            xmin *= scale_x
            xmax *= scale_x
            ymin *= scale_y
            ymax *= scale_y
            depth_cm = data[4]
            quat = torch.tensor(data[5:], dtype=torch.float32)

        img_cropped = img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
        img_tensor = self.transform(img_cropped)  # apply fixed resize + ToTensor

        z = torch.tensor([depth_cm], dtype=torch.float32)

        obj_class = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        if obj_class not in CLASS_MAPPING:
            raise ValueError(f"[ERROR] Object class '{obj_class}' not found in CLASS_MAPPING. Path: {img_path}")

        return img_tensor, z, quat, os.path.basename(img_path)


class PoseResNet18(nn.Module):
    def __init__(self):
        super(PoseResNet18, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 5)

    def forward(self, x):
        return self.backbone(x)

def draw_pose_axes(img, quat, bbox=None, color_x=(0,0,255), color_y=(0,255,0), color_z=(255,0,0), length=80):
    """
    Dibuja sistema de coordenadas x,y,z en el centro de la imagen recortada, orientado por quaternion.
    Si se pasa bbox=(xmin,ymin,xmax,ymax), dibuja en centro del bbox en img original.
    """
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    if bbox is not None:
        cx = (bbox[0]+bbox[2])/2
        cy = (bbox[1]+bbox[3])/2
    else:
        w,h = img.size
        cx, cy = w//2, h//2

    rot = R.from_quat(quat.cpu().numpy())
    axes = rot.apply(np.eye(3))

    for axis, color in zip(axes, [color_x, color_y, color_z]):
        end_x = cx + axis[0] * length
        end_y = cy - axis[1] * length  # Y invertido imagen
        draw.line([(cx, cy), (end_x, end_y)], fill=color, width=5)

    return img_draw


def save_pose_example_outputs_pose(imgs, quat_preds, quat_gts, img_names, out_path, scene):
    """
    Guarda un ejemplo por objeto para orientation_88_-6_-34:
    imagen original, predicha con ejes XYZ, ground truth con ejes XYZ.
    Usa index 60% si scene==2, sino usa la mitad.
    """
    out_dir = Path(out_path) / "examples"
    out_dir.mkdir(exist_ok=True)

    orientation_objects = defaultdict(list)

    # Agrupar indices por objeto
    for i, name in enumerate(img_names):
        if "orientation_88_-6_-34" in name:
            obj = name.split("/")[-3]
            orientation_objects[obj].append((i, name))

    # Procesar cada objeto
    for obj in sorted(orientation_objects.keys()):
        entries = orientation_objects[obj]
        sorted_entries = sorted(entries, key=lambda x: os.path.basename(x[1]))

        indices_sorted = [i for i, _ in sorted_entries]
        if not indices_sorted:
            continue

        if int(scene) == 2:
            idx = indices_sorted[int(0.6 * len(indices_sorted))]
        else:
            idx = indices_sorted[len(indices_sorted)//2]

        img_tensor = imgs[idx].cpu()
        quat_pred = quat_preds[idx].cpu()
        quat_gt = quat_gts[idx].cpu()
        img_name = os.path.basename(img_names[idx])

        save_pose_example_outputs(img_tensor, quat_pred, quat_gt, out_dir, img_name)

def quaternion_angle_error(q_pred, q_gt):
    q_pred = q_pred / q_pred.norm(dim=-1, keepdim=True)
    q_gt = q_gt / q_gt.norm(dim=-1, keepdim=True)
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=-1))
    dot = torch.clamp(dot, -1.0, 1.0)
    angle = 2 * torch.acos(dot) * (180.0 / np.pi)
    return angle

def quaternion_to_euler_deg(quat):
    r = R.from_quat(quat.cpu().numpy())
    euler_deg = r.as_euler('xyz', degrees=True)
    return torch.tensor(euler_deg)



def train_eval(args, model, device, train_loader, val_loader):
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    out_path = os.path.join(args.output_dir, f"{args.sensor}_scene_{args.scene}")
    os.makedirs(out_path, exist_ok=True)

    csv_log_path = os.path.join(out_path, 'metrics.csv')
    with open(csv_log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_total_loss', 'train_z_loss', 'train_q_loss',
                         'val_total_loss', 'val_z_loss', 'val_q_loss',
                         'train_mae_z', 'train_q_mse', 'train_q_angle',
                         'val_mae_z', 'val_q_mse', 'val_q_angle',
                         'val_roll_error', 'val_pitch_error', 'val_yaw_error'])

        for epoch in range(args.epochs):
            model.train()
            train_total_loss = train_z_loss = train_q_loss = 0
            train_mae_z = train_q_mse = train_q_angle = 0

            for images, z, quat, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
                images, z, quat = images.to(device), z.to(device), quat.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                z_pred, quat_pred = outputs[:,0], outputs[:,1:]

                z_loss = criterion(z_pred, z.squeeze())
                q_loss = criterion(quat_pred, quat)
                loss = z_loss + q_loss
                loss.backward()
                optimizer.step()

                train_total_loss += loss.item()
                train_z_loss += z_loss.item()
                train_q_loss += q_loss.item()

                train_mae_z += torch.mean(torch.abs(z_pred - z.squeeze())).item()
                train_q_mse += torch.mean((quat_pred - quat)**2).item()
                train_q_angle += torch.mean(quaternion_angle_error(quat_pred, quat)).item()

            n_train = len(train_loader)
            tl, zl, ql = train_total_loss/n_train, train_z_loss/n_train, train_q_loss/n_train
            mae_z_epoch = train_mae_z / n_train
            q_mse_epoch = train_q_mse / n_train
            q_angle_epoch = train_q_angle / n_train

            model.eval()
            val_total_loss = val_z_loss = val_q_loss = 0
            val_mae_z = val_q_mse = val_q_angle = 0
            val_roll_error = val_pitch_error = val_yaw_error = 0

            images_all, quat_preds_all, quat_gts_all, img_names_all = [], [], [], []

            with torch.no_grad():
                for images, z, quat, img_names in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                    images, z, quat = images.to(device), z.to(device), quat.to(device)
                    outputs = model(images)
                    z_pred, quat_pred = outputs[:,0], outputs[:,1:]

                    z_loss_each = (z_pred - z.squeeze()).pow(2)
                    q_loss_each = (quat_pred - quat).pow(2).mean(dim=1)
                    loss_each = z_loss_each + q_loss_each

                    z_mae_each = torch.abs(z_pred - z.squeeze())
                    q_mse_each = (quat_pred - quat).pow(2).mean(dim=1)
                    q_angle_each = quaternion_angle_error(quat_pred, quat)

                    euler_pred = quaternion_to_euler_deg(quat_pred)
                    euler_gt = quaternion_to_euler_deg(quat)
                    euler_error_each = torch.abs(euler_pred - euler_gt)

                    val_total_loss += loss_each.mean().item()
                    val_z_loss += z_loss_each.mean().item()
                    val_q_loss += q_loss_each.mean().item()
                    val_mae_z += z_mae_each.mean().item()
                    val_q_mse += q_mse_each.mean().item()
                    val_q_angle += q_angle_each.mean().item()
                    val_roll_error += euler_error_each[:,0].mean().item()
                    val_pitch_error += euler_error_each[:,1].mean().item()
                    val_yaw_error += euler_error_each[:,2].mean().item()

                    images_all.extend(images.cpu())
                    quat_preds_all.extend(quat_pred.cpu())
                    quat_gts_all.extend(quat.cpu())
                    img_names_all.extend(img_names)

            n_val = len(val_loader)
            vl, vzl, vql = val_total_loss/n_val, val_z_loss/n_val, val_q_loss/n_val
            val_mae_z_epoch = val_mae_z / n_val
            val_q_mse_epoch = val_q_mse / n_val
            val_q_angle_epoch = val_q_angle / n_val
            val_roll_error_epoch = val_roll_error / n_val
            val_pitch_error_epoch = val_pitch_error / n_val
            val_yaw_error_epoch = val_yaw_error / n_val

            writer.writerow([epoch+1, tl, zl, ql, vl, vzl, vql,
                             mae_z_epoch, q_mse_epoch, q_angle_epoch,
                             val_mae_z_epoch, val_q_mse_epoch, val_q_angle_epoch,
                             val_roll_error_epoch, val_pitch_error_epoch, val_yaw_error_epoch])


            print(f"[INFO] Epoch {epoch+1} | Train Loss: {tl:.4f}, Val Loss: {vl:.4f}, "
                  f"MAE_z: {val_mae_z_epoch:.2f}cm, "
                  f"Angle_err: {val_q_angle_epoch:.1f}deg, "  # <-- añade esta línea
                  f"Roll: {val_roll_error_epoch:.1f}, "
                  f"Pitch: {val_pitch_error_epoch:.1f}, "
                  f"Yaw: {val_yaw_error_epoch:.1f}")


            save_pose_example_outputs_pose(images_all, quat_preds_all, quat_gts_all, img_names_all, out_path, args.scene)
            torch.save(model.state_dict(), os.path.join(out_path, f'model_epoch{epoch+1}.pth'))





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor', required=True, choices=["asus", "davis346", "evk4"])
    parser.add_argument('--scene', required=True, type=int, choices=[0,1,2,3])
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=int, default=0, help="GPU index to use (default: 0)")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # === Dataset y DataLoader ===
    train_dataset, val_dataset = get_datasets(args.sensor, args.input_dir, args.scene)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    # === Modelo y device ===
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = PoseResNet18().to(device)

    train_eval(args, model, device, train_loader, val_loader)




