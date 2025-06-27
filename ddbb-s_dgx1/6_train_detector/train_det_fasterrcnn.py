# file: train_fasterrcnn_full.py

import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import csv


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


class FasterRCNNDataset(Dataset):
    def __init__(self, image_paths, bbox_paths, sensor, transforms=None):
        self.image_paths = image_paths
        self.bbox_paths = bbox_paths
        self.sensor = sensor
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        orig_w, orig_h = img.size

        # === Resize si es evk4 ===
        if "evk4" in img_path:
            resized_h, resized_w = SENSOR_SHAPES["evk4"][0] // 2, SENSOR_SHAPES["evk4"][1] // 2
            img = img.resize((resized_w, resized_h), Image.BILINEAR)
            scale_x = resized_w / orig_w
            scale_y = resized_h / orig_h
        else:
            scale_x = scale_y = 1.0

        # === Bboxes ===
        bbox_path = self.bbox_paths[idx]
        with open(bbox_path) as f:
            lines = f.readlines()[1:]  # skip header
            bboxes = []
            for line in lines:
                xmin, ymin, xmax, ymax = map(float, line.strip().split(','))
                # === Aplicar escala si es evk4 ===
                xmin *= scale_x
                xmax *= scale_x
                ymin *= scale_y
                ymax *= scale_y
                bboxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(bboxes, dtype=torch.float32)

        # === Clase desde carpeta padre ===
        obj_class = os.path.basename(os.path.dirname(img_path))
        label = CLASS_MAPPING.get(obj_class, 0)
        labels = torch.tensor([label for _ in boxes], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }

        if self.transforms:
            img = self.transforms(img)
        else:
            img = transforms.ToTensor()(img)

        return img, target, img_path


def load_paths_det(data_txt, bbox_txt):
    with open(data_txt) as f:
        image_paths = [line.strip() for line in f]
    with open(bbox_txt) as f:
        bbox_paths = [line.strip() for line in f]
    return image_paths, bbox_paths

def get_fasterrcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def save_example_outputs(imgs, preds, targets, paths, out_path):
    out_dir = Path(out_path) / "examples"
    out_dir.mkdir(exist_ok=True)

    orientation_objects = defaultdict(list)

    # === Primer loop: agrupar (i, path) por objeto ===
    for i, path in enumerate(paths):
        if "orientation_88_-6_-34" in str(path):
            obj = str(path).split("/")[-3]
            orientation_objects[obj].append((i, path))

    # === Segundo loop: procesar cada objeto ===
    for obj in sorted(orientation_objects.keys()):
        entries = orientation_objects[obj]
        sorted_entries = sorted(entries, key=lambda x: os.path.basename(str(x[1])))

        indices_sorted = [i for i, _ in sorted_entries]
        if not indices_sorted:
            continue

        # === Selección index 60% o mitad ===
        if "scene_2" in str(out_path):
            idx = indices_sorted[int(0.6 * len(indices_sorted))]
        else:
            idx = indices_sorted[len(indices_sorted)//2]

        # === Recuperar imágenes y predicciones ===
        img = imgs[idx].cpu()
        img_pil = transforms.ToPILImage()(img)
        pred = preds[idx]
        target = targets[idx]

        # === Imagen con predicciones ===
        pred_img = img_pil.copy()
        draw_pred = ImageDraw.Draw(pred_img)

        if len(pred['boxes']) > 0:
            for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                draw_pred.rectangle(list(box), outline="red", width=2)
                draw_pred.text((box[0], box[1]), f"{label.item()}:{score:.2f}", fill="red")

        # === Imagen con ground truth ===
        gt_img = img_pil.copy()
        draw_gt = ImageDraw.Draw(gt_img)
        for box, label in zip(target['boxes'], target['labels']):
            draw_gt.rectangle(list(box), outline="green", width=2)
            draw_gt.text((box[0], box[1]), str(label.item()), fill="green")

        # === Concatenar vertical ===
        total_height = img_pil.height * 3
        max_width = img_pil.width

        concatenated = Image.new("RGB", (max_width, total_height))
        concatenated.paste(img_pil, (0,0))
        concatenated.paste(pred_img, (0, img_pil.height))
        concatenated.paste(gt_img, (0, img_pil.height*2))

        concatenated.save(out_dir / f"example_{obj}_orientation_88_-6_-34.png")



def train_eval(model, dataset, device, out_path, num_epochs=10, lr=1e-4):
    data_loader = DataLoader(
        dataset, batch_size=4, shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    metrics_csv = os.path.join(out_path, "metrics.csv")
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "mAP_50", "mAP_95"])

    best_map95 = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for imgs, targets, paths in tqdm(data_loader, desc=f"Epoch {epoch+1} [Train]"):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

        print(f"[INFO] Epoch {epoch+1} Loss: {epoch_loss:.4f}")

        # === Evaluación COCO ===
        model.eval()
        coco_gt, coco_dt, coco_images = [], [], []
        annotation_id = 1

        # === Acumuladores globales para ejemplos ===
        all_imgs, all_preds, all_targets, all_paths = [], [], [], []

        with torch.no_grad():
            for idx, (imgs, targets, paths_batch) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1} [Val]")):
                imgs = [img.to(device) for img in imgs]
                outputs = model(imgs)

                for t, o, img, path in zip(targets, outputs, imgs, paths_batch):
                    image_id = int(t["image_id"].item())

                    # === Ground Truth ===
                    for box, label in zip(t['boxes'], t['labels']):
                        xmin, ymin, xmax, ymax = box.cpu().numpy().tolist()
                        coco_gt.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [xmin, ymin, xmax-xmin, ymax-ymin],
                            "area": (xmax-xmin)*(ymax-ymin),
                            "iscrowd": 0
                        })
                        annotation_id += 1

                    # === Predictions ===
                    for box, score, label in zip(o['boxes'], o['scores'], o['labels']):
                        xmin, ymin, xmax, ymax = box.cpu().numpy().tolist()
                        coco_dt.append({
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [xmin, ymin, xmax-xmin, ymax-ymin],
                            "score": float(score)
                        })

                    # === Images ===
                    coco_images.append({
                        "id": image_id,
                        "width": img.shape[2],
                        "height": img.shape[1]
                    })

                # === Acumular para ejemplos ===
                all_imgs.extend([img.cpu() for img in imgs])
                all_preds.extend(outputs)
                all_targets.extend(targets)
                all_paths.extend(paths_batch)

        # === COCO Evaluation structures ===
        coco_gt_dict = {
            "images": coco_images,
            "annotations": coco_gt,
            "categories": [{"id": v, "name": k} for k, v in CLASS_MAPPING.items()]
        }

        # === Save intermediate jsons
        with open(os.path.join(out_path, "coco_gt.json"), 'w') as f:
            json.dump(coco_gt_dict, f)
        with open(os.path.join(out_path, "coco_dt.json"), 'w') as f:
            json.dump(coco_dt, f)

        coco_gt_obj = COCO(os.path.join(out_path, "coco_gt.json"))
        coco_dt_obj = coco_gt_obj.loadRes(os.path.join(out_path, "coco_dt.json"))

        coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # === Extraer mAP@50 y mAP@95 ===
        mAP_50 = coco_eval.stats[1]  # AP IoU=0.50
        mAP_95 = coco_eval.stats[0]  # AP IoU=0.50:0.95

        # === Guardar en metrics.csv ===
        with open(metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, epoch_loss, mAP_50, mAP_95])

        # === Guardar en metrics.txt ===
        with open(os.path.join(out_path, "metrics.txt"), 'w') as f:
            f.write(f"Epoch {epoch+1}\n")
            f.write(f"Train Loss: {epoch_loss:.4f}\n")
            f.write(f"mAP@50: {mAP_50:.4f}\n")
            f.write(f"mAP@95: {mAP_95:.4f}\n\n")

            # === Metrics by class ===
            precisions = coco_eval.eval['precision']
            for idx, cat in enumerate(CLASS_MAPPING.keys()):
                precision_cat = precisions[:, idx, 0, 0, -1]
                precision_cat = precision_cat[precision_cat > -1]
                ap_cat = np.mean(precision_cat) if len(precision_cat) else float('nan')
                f.write(f"{cat}: AP={ap_cat:.4f}\n")

        # === Guardar mejor modelo y ejemplos ===
        if mAP_95 > best_map95:
            best_map95 = mAP_95
            torch.save(model.state_dict(), os.path.join(out_path, "fasterrcnn_model.pth"))

            save_example_outputs(all_imgs, all_preds, all_targets, all_paths, out_path)

    print(f"[DONE] Best model saved to {out_path}/fasterrcnn_model.pth")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor', required=True, choices=["asus", "davis346", "evk4"])
    parser.add_argument('--scene', required=True, type=int, choices=[0,1,2,3])
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    data_txt = os.path.join(args.input_dir, f"{args.sensor}_data_scene_{args.scene}.txt")
    bbox_txt = os.path.join(args.input_dir, f"{args.sensor}_det-bbox-abs_scene_{args.scene}.txt")
    image_paths, bbox_paths = load_paths_det(data_txt, bbox_txt)

    dataset = FasterRCNNDataset(image_paths, bbox_paths, args.sensor, transforms=transforms.ToTensor())
    model = get_fasterrcnn_model(num_classes=len(CLASS_MAPPING)+1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_eval(model, dataset, device, args.output_dir, num_epochs=args.epochs, lr=args.lr)
