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
import torch.nn.functional as F
import yaml


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

            x_px = data[4]
            y_px = data[5]


            # x_px, y_px son relativos a la imagen original
            # Convertirlos a coordenadas relativas al crop:

            x_px_crop = x_px - xmin
            y_px_crop = y_px - ymin

            crop_w = xmax - xmin
            crop_h = ymax - ymin

            # Normalizar en [0,1] respecto al crop
            x_px_norm = x_px_crop / crop_w
            y_px_norm = y_px_crop / crop_h

            x_px = torch.tensor([x_px_norm], dtype=torch.float32)
            y_px = torch.tensor([y_px_norm], dtype=torch.float32)

            depth_cm = data[6]
            quat = torch.tensor(data[7:11], dtype=torch.float32)
            quat = quat / quat.norm()

        img_cropped = img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
        img_tensor = self.transform(img_cropped)  # apply fixed resize + ToTensor

        z = torch.tensor([depth_cm], dtype=torch.float32)

        obj_class = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        if obj_class not in CLASS_MAPPING:
            raise ValueError(f"[ERROR] Object class '{obj_class}' not found in CLASS_MAPPING. Path: {img_path}")

        crop_w = torch.tensor([crop_w], dtype=torch.float32)
        crop_h = torch.tensor([crop_h], dtype=torch.float32)
        xmin = torch.tensor([xmin], dtype=torch.float32)
        ymin = torch.tensor([ymin], dtype=torch.float32)

        return img_tensor, x_px, y_px, z, quat, img_path, crop_w, crop_h, xmin, ymin



class PoseResNet50TwoHeads(nn.Module):
    def __init__(self):
        super(PoseResNet50TwoHeads, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        num_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove original FC

        self.head_pos = nn.Linear(num_feats, 3)   # x,y,z
        self.head_quat = nn.Linear(num_feats, 4)  # q0,q1,q2,q3

    def forward(self, x):
        feats = self.backbone(x)
        pos = self.head_pos(feats)
        quat = self.head_quat(feats)
        quat = F.normalize(quat, dim=-1)  # ensure unit quaternion
        return pos, quat


def draw_pose_axes(img, quat=None, cx=None, cy=None, bbox=None,
                   color_x=(255,0,0), color_y=(0,255,0), color_z=(0,0,255),
                   length=120):

    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    # === Centro de referencia ===
    if cx is not None and cy is not None:
        cx, cy = cx, cy
    elif bbox is not None:
        cx = (bbox[0]+bbox[2])/2
        cy = (bbox[1]+bbox[3])/2
    else:
        w,h = img.size
        cx, cy = w//2, h//2

    if quat is not None:
        # === Reconstrucción de orientación del objeto como en script 3D ===
        rot_rel = R.from_quat(quat)
        rot_cam = R.from_euler('x', 90, degrees=True)
        rot_obj = rot_cam * rot_rel
        rot_extra_x = R.from_euler('x', 0, degrees=True)
        rot_obj_adjusted = rot_obj * rot_extra_x

        # === Obtener ejes transformados en WORLD ===
        axes_obj = rot_obj_adjusted.apply(np.eye(3))

        # === Proyección 2D ===
        for axis_vec, color in zip(axes_obj, [color_x, color_y, color_z]):
            dx = axis_vec[0]
            dy = axis_vec[2]  # usar Z como eje vertical en imagen

            end_x = cx + dx * length
            end_y = cy - dy * length  # PIL: +Y abajo

            draw.line([(cx, cy), (end_x, end_y)], fill=color, width=5)

    return img_draw


def save_pose_example_outputs(img_tensor, x_px_pred, y_px_pred, x_px_gt, y_px_gt, quat_pred, quat_gt, out_dir, img_name):
    """
    Guarda una imagen concatenada verticalmente:
    - Original
    - Predicha con ejes XYZ dibujados según quat_pred
    - Ground truth con ejes XYZ dibujados según quat_gt
    """
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(img_tensor)

    # Convertir coordenadas si son tensors
    x_px_pred = x_px_pred.item() if isinstance(x_px_pred, torch.Tensor) else x_px_pred
    y_px_pred = y_px_pred.item() if isinstance(y_px_pred, torch.Tensor) else y_px_pred
    x_px_gt = x_px_gt.item() if isinstance(x_px_gt, torch.Tensor) else x_px_gt
    y_px_gt = y_px_gt.item() if isinstance(y_px_gt, torch.Tensor) else y_px_gt

    # Escalar a pixeles del crop
    width, height = img_pil.size
    x_px_pred_pix = x_px_pred * width
    y_px_pred_pix = y_px_pred * height
    x_px_gt_pix = x_px_gt * width
    y_px_gt_pix = y_px_gt * height

    # Imagen original
    img_original = img_pil.copy()

    # Imagen predicha
    img_pred = draw_pose_axes(img_pil, quat_pred, cx=x_px_pred_pix, cy=y_px_pred_pix)

    # Imagen ground truth
    img_gt = draw_pose_axes(img_pil, quat_gt, cx=x_px_gt_pix, cy=y_px_gt_pix)

    # Concatenar verticalmente
    concatenated = Image.new('RGB', (width, height * 3))
    concatenated.paste(img_original, (0, 0))
    concatenated.paste(img_pred, (0, height))
    concatenated.paste(img_gt, (0, height * 2))

    # Guardar
    concatenated.save(out_dir / f"{img_name}_concat.png")




def save_pose_example_outputs_pose(imgs, x_preds, y_preds, x_gts, y_gts,
                                   quat_preds, quat_gts, img_names, out_path, scene):
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

        save_pose_example_outputs(
            img_tensor,
            x_preds[idx].cpu(), y_preds[idx].cpu(),
            x_gts[idx].cpu(), y_gts[idx].cpu(),
            quat_preds[idx].cpu(), quat_gts[idx].cpu(),
            out_dir, img_name
        )

def quaternion_angle_error(q_pred, q_gt):
    q_pred = q_pred / q_pred.norm(dim=-1, keepdim=True)
    q_gt = q_gt / q_gt.norm(dim=-1, keepdim=True)
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=-1))
    dot = torch.clamp(dot, -1.0, 1.0)
    angle = 2 * torch.acos(dot) * (180.0 / np.pi)
    return angle

def quaternion_to_euler_deg(quat):
    rot_rel = R.from_quat(quat.cpu().numpy())
    rot_cam = R.from_euler('x', 90, degrees=True)
    rot_obj = rot_cam * rot_rel
    rot_extra_x = R.from_euler('x', -90, degrees=True)
    rot_obj_adjusted = rot_obj * rot_extra_x
    euler_deg = rot_obj_adjusted.as_euler('xyz', degrees=True)
    return torch.tensor(euler_deg)

def geodesic_loss(q_pred, q_gt):
    """
    Calcula el error angular entre quaterniones predicho y ground truth.
    Retorna el error en radianes.
    """
    q_pred = q_pred / q_pred.norm(dim=-1, keepdim=True)
    q_gt = q_gt / q_gt.norm(dim=-1, keepdim=True)
    dot = torch.sum(q_pred * q_gt, dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0)
    theta = 2 * torch.acos(torch.abs(dot))  # en radianes
    return theta.mean()  # promedio en el batch


def train_eval(args, model, device, train_loader, val_loader):
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    with open(f"sensors/config_{args.sensor}.yaml") as f:
        cam_config = yaml.safe_load(f)
    focal_length_px = cam_config['camera']['focal_length_px']
    focal_length_py = cam_config['camera']['focal_length_py']
    res_w = cam_config['camera']['resolution']['width']
    res_h = cam_config['camera']['resolution']['height']

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    out_path = os.path.join(args.output_dir, f"{args.sensor}_scene_{args.scene}")
    os.makedirs(out_path, exist_ok=True)

    csv_log_path = os.path.join(out_path, 'metrics.csv')
    per_object_dir = os.path.join(out_path, 'per_object_metrics')
    os.makedirs(per_object_dir, exist_ok=True)

    with open(csv_log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'val_total_loss', 'val_pos_loss', 'val_q_geodesic',
                         'val_mae_x', 'val_mae_y', 'val_mae_z',
                         'err_x_cm', 'err_y_cm',
                         'val_q_mse', 'val_q_angle',
                         'val_roll_error', 'val_pitch_error', 'val_yaw_error'])

        for epoch in range(args.epochs):
            model.train()
            for images, x_px, y_px, z, quat, *_ in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):

                images, x_px, y_px, z, quat = images.to(device), x_px.to(device), y_px.to(device), z.to(device), quat.to(device)
                optimizer.zero_grad()
                
                pos_pred, quat_pred = model(images)
                pos_gt = torch.cat([x_px, y_px, z], dim=1).squeeze()

                pos_loss = criterion(pos_pred, pos_gt)
                q_loss = geodesic_loss(quat_pred, quat)  # ahora q_loss es el error angular medio en radianes
                loss = pos_loss + 50 * q_loss  # el factor depende de la magnitud; ajusta tras observar entrenamiento


                loss.backward()
                optimizer.step()


            model.eval()
            val_metrics = defaultdict(lambda: defaultdict(float))
            val_counts = defaultdict(int)
            
            images_all, quat_preds_all, quat_gts_all, img_names_all = [], [], [], []
            x_px_preds_all, y_px_preds_all, x_px_gts_all, y_px_gts_all = [], [], [], []
            err_x_cm_all = []
            err_y_cm_all = []

            total_q_geodesic = 0.0
            num_batches = 0

            with torch.no_grad():
                for images, x_px, y_px, z, quat, img_names, crop_w, crop_h, xmin, ymin in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):

                    images, x_px, y_px, z, quat = images.to(device), x_px.to(device), y_px.to(device), z.to(device), quat.to(device)
                    pos_pred, quat_pred = model(images)

                    pos_gt = torch.cat([x_px, y_px, z], dim=1).squeeze()
                    pos_loss_batch = criterion(pos_pred, pos_gt).item()
                    q_loss_batch = geodesic_loss(quat_pred, quat).item()

                    total_q_geodesic += q_loss_batch
                    num_batches += 1


                    # Resto de métricas permanece igual
                    x_px_pred, y_px_pred, z_pred = pos_pred[:,0], pos_pred[:,1], pos_pred[:,2]


                    crop_w = crop_w.to(x_px_pred.device)
                    crop_h = crop_h.to(x_px_pred.device)
                    xmin = xmin.to(x_px_pred.device)
                    ymin = ymin.to(x_px_pred.device)

                    # Convertir x_px_pred y x_px_gt de normalizado [0,1] a px absolutos en crop
                    x_px_pred_abs = x_px_pred * crop_w + xmin
                    y_px_pred_abs = y_px_pred * crop_h + ymin

                    x_px_gt_abs = x_px.squeeze() * crop_w + xmin
                    y_px_gt_abs = y_px.squeeze() * crop_h + ymin

                    # Reproyección a cm usando pinhole (Z = profundidad en cm)
                    # Delta_u * Z / f
                    delta_x_px = x_px_pred_abs - x_px_gt_abs
                    delta_y_px = y_px_pred_abs - y_px_gt_abs

                    # Ajusta si focal length está en mm. Si tu focal_length_px ya es en px según la calibración, usa directamente:
                    err_x_cm = delta_x_px * z_pred / focal_length_px
                    err_y_cm = delta_y_px * z_pred / focal_length_py

                    # Guarda abs para métricas globales
                    err_x_cm_abs = torch.abs(err_x_cm).cpu().numpy()
                    err_y_cm_abs = torch.abs(err_y_cm).cpu().numpy()

                    err_x_cm_all.extend(err_x_cm_abs.tolist())
                    err_y_cm_all.extend(err_y_cm_abs.tolist())

                    x_mae_each = torch.abs(x_px_pred - x_px.squeeze()).cpu().numpy()
                    y_mae_each = torch.abs(y_px_pred - y_px.squeeze()).cpu().numpy()
                    z_mae_each = torch.abs(z_pred - z.squeeze()).cpu().numpy()
                    q_mse_each = ((quat_pred - quat)**2).mean(dim=1).cpu().numpy()
                    q_angle_each = quaternion_angle_error(quat_pred, quat).cpu().numpy()

                    euler_pred = quaternion_to_euler_deg(quat_pred)
                    euler_gt = quaternion_to_euler_deg(quat)
                    euler_err = torch.abs(euler_pred - euler_gt).cpu().numpy()

                    for i, name in enumerate(img_names):
                        obj_name = name.split("/")[-3]
                        val_counts[obj_name] += 1
                        val_metrics[obj_name]['x_mae'] += x_mae_each[i]
                        val_metrics[obj_name]['y_mae'] += y_mae_each[i]
                        val_metrics[obj_name]['z_mae'] += z_mae_each[i]
                        val_metrics[obj_name]['q_mse'] += q_mse_each[i]
                        val_metrics[obj_name]['q_angle'] += q_angle_each[i]
                        val_metrics[obj_name]['roll'] += euler_err[i,0]
                        val_metrics[obj_name]['pitch'] += euler_err[i,1]
                        val_metrics[obj_name]['yaw'] += euler_err[i,2]

                    images_all.extend(images.cpu())
                    quat_preds_all.extend(quat_pred.cpu())
                    quat_gts_all.extend(quat.cpu())
                    img_names_all.extend(img_names)
                    x_px_preds_all.extend(x_px_pred.cpu())
                    y_px_preds_all.extend(y_px_pred.cpu())
                    x_px_gts_all.extend(x_px.cpu())
                    y_px_gts_all.extend(y_px.cpu())

            val_q_geodesic = total_q_geodesic / num_batches


            # Guardar métricas por objeto (archivo separado por objeto)
            for obj, metrics in val_metrics.items():
                count = val_counts[obj]
                obj_csv = os.path.join(per_object_dir, f"{obj}.csv")
                with open(obj_csv, 'w', newline='') as fobj:
                    writer_obj = csv.writer(fobj)
                    writer_obj.writerow(['epoch', 'x_mae', 'y_mae', 'z_mae', 'q_mse', 'q_angle', 'roll', 'pitch', 'yaw'])
                    writer_obj.writerow([epoch+1,
                                        metrics['x_mae']/count,
                                        metrics['y_mae']/count,
                                        metrics['z_mae']/count,
                                        metrics['q_mse']/count,
                                        metrics['q_angle']/count,
                                        metrics['roll']/count,
                                        metrics['pitch']/count,
                                        metrics['yaw']/count])

            # Guardar modelo por epoch
            torch.save(model.state_dict(), os.path.join(out_path, f'model.pth'))

            # Guardar métricas globales en csv principal
            total_count = sum(val_counts.values())
            val_mae_x = sum(m['x_mae'] for m in val_metrics.values()) / total_count
            val_mae_y = sum(m['y_mae'] for m in val_metrics.values()) / total_count
            val_mae_z = sum(m['z_mae'] for m in val_metrics.values()) / total_count
            val_q_mse = sum(m['q_mse'] for m in val_metrics.values()) / total_count
            val_q_angle = sum(m['q_angle'] for m in val_metrics.values()) / total_count
            val_roll = sum(m['roll'] for m in val_metrics.values()) / total_count
            val_pitch = sum(m['pitch'] for m in val_metrics.values()) / total_count
            val_yaw = sum(m['yaw'] for m in val_metrics.values()) / total_count

            val_pos_loss = (val_mae_x + val_mae_y + val_mae_z) / 3
            total_loss = val_pos_loss + 100 * val_q_geodesic

            mean_err_x_cm = np.mean(err_x_cm_all)
            mean_err_y_cm = np.mean(err_y_cm_all)


            writer.writerow([epoch+1,
                             total_loss, val_pos_loss, val_q_geodesic,
                             val_mae_x, val_mae_y, val_mae_z,
                             mean_err_x_cm, mean_err_y_cm,
                             val_q_mse, val_q_angle,
                             val_roll, val_pitch, val_yaw])

            # Guardar ejemplos de outputs con save_pose_example_outputs_pose
            save_pose_example_outputs_pose(
                images_all,
                x_px_preds_all, y_px_preds_all,   # ← NUEVO: listas con x/y predichos
                x_px_gts_all, y_px_gts_all,       # ← NUEVO: listas con x/y ground truth
                quat_preds_all, quat_gts_all,
                img_names_all,
                out_path, args.scene
            )

            print(f"[VAL] Epoch {epoch+1}: x_mae={val_mae_x:.3f}, y_mae={val_mae_y:.3f}, z_mae={val_mae_z:.1f}, "
                  f"err_x_cm={mean_err_x_cm:.2f} cm, err_y_cm={mean_err_y_cm:.2f} cm, "
                  f"q_mse={val_q_mse:.4f}, q_angle={val_q_angle:.1f} deg, "
                  f"roll_err={val_roll:.1f}, pitch_err={val_pitch:.1f}, yaw_err={val_yaw:.1f}")


            print(f"[INFO] Epoch {epoch+1} completado. Model, métricas globales, por objeto y ejemplos guardados.")

    print("✅ Entrenamiento finalizado con modelos guardados por epoch, métricas globales + por objeto y ejemplos.")


# ✅ Incluye x/y losses en csv y en logs terminal, listo para integrarlo a tu training pipeline.




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
    model = PoseResNet50TwoHeads().to(device)


    train_eval(args, model, device, train_loader, val_loader)




