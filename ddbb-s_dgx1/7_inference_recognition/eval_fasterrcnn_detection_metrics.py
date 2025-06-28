"""
file: eval_fasterrcnn_metrics.py
Purpose: Updated to exclude 'arbol' completely from evaluation metrics and confusion matrix.
"""

import os
import argparse
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

class FasterRCNNDataset(Dataset):
    def __init__(self, image_paths, bbox_paths, tfms=None):
        self.image_paths = image_paths
        self.bbox_paths = bbox_paths
        self.transforms = tfms or transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        with open(self.bbox_paths[idx]) as f:
            bboxes = [list(map(float, line.strip().split(','))) for line in f.readlines()[1:]]
        boxes = torch.tensor(bboxes, dtype=torch.float32)
        obj_class = os.path.basename(os.path.dirname(os.path.dirname(self.image_paths[idx])))
        label = CLASS_MAPPING.get(obj_class, 0)
        labels = torch.full((len(boxes),), label, dtype=torch.int64)
        return self.transforms(img), {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

def load_paths(data_txt, bbox_txt):
    with open(data_txt) as f:
        image_paths = [line.strip() for line in f]
    with open(bbox_txt) as f:
        bbox_paths = [line.strip() for line in f]
    return image_paths, bbox_paths

def run_evaluation(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_path = os.path.join(args.output_dir, f"{args.sensor}_scene_{args.scene}_eval")
    os.makedirs(out_path, exist_ok=True)

    data_txt = os.path.join(args.input_dir, f"{args.sensor}_data_scene_{args.scene}.txt")
    bbox_txt = os.path.join(args.input_dir, f"{args.sensor}_det-bbox-abs-10ms_scene_{args.scene}.txt")
    image_paths, bbox_paths = load_paths(data_txt, bbox_txt)

    val_imgs, val_bboxes = zip(*[(i,b) for i,b in zip(image_paths, bbox_paths) if any(o in i for o in VAL_ORIENTATIONS)])
    val_loader = DataLoader(FasterRCNNDataset(val_imgs, val_bboxes), batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = fasterrcnn_resnet50_fpn(weights=None)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, len(CLASS_MAPPING)+1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            outputs = model([img.to(device) for img in imgs])
            for t, o in zip(targets, outputs):
                gt = t['labels'].cpu().numpy().tolist()
                pred = [l for l,s in zip(o['labels'].cpu().numpy(), o['scores'].cpu().numpy()) if s>0.5]
                y_true.extend(gt)
                y_pred.extend(pred if pred else [0])

    # === Exclude 'arbol' ===
    filtered = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yt != 2 and yp != 2]
    y_true_f, y_pred_f = zip(*filtered) if filtered else ([],[])

    # === Remove 'arbol' from CLASS_MAPPING ===
    class_mapping_filtered = {k:v for k,v in CLASS_MAPPING.items() if k != 'arbol'}

    labels_filtered = list(class_mapping_filtered.values())
    target_names_filtered = list(class_mapping_filtered.keys())

    report = pd.DataFrame(classification_report(y_true_f, y_pred_f, labels=labels_filtered, target_names=target_names_filtered, zero_division=0, output_dict=True)).transpose()
    report.to_csv(os.path.join(out_path, "classification_report.csv"))

    acc = accuracy_score(y_true_f, y_pred_f)
    with open(os.path.join(out_path, "metrics.txt"), 'w') as f:
        f.write(f"Global Accuracy: {acc:.4f}\n")

    cm = confusion_matrix(y_true_f, y_pred_f, labels=labels_filtered)
    np.savetxt(os.path.join(out_path, "confusion_matrix.txt"), cm, fmt='%d')

    print(f"[DONE] Metrics saved to {out_path}")

if __name__ == "__main__":
    CLASS_MAPPING = {
        'almohada': 1, 'arbol': 2, 'avion': 3, 'boomerang': 4, 'caja_amarilla': 5,
        'caja_azul': 6, 'carro_rojo': 7, 'clorox': 8, 'dino': 9, 'jarron': 10,
        'lysoform': 11, 'mobil': 12, 'paleta': 13, 'pelota': 14, 'sombrero': 15,
        'tarro': 16, 'zapatilla': 17
    }

    VAL_ORIENTATIONS = [
        "orientation_39_17_-102", "orientation_19_31_21",
        "orientation_-125_66_-116", "orientation_88_-6_-34"
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor', required=True)
    parser.add_argument('--scene', type=int, required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_evaluation(args)

"""
# Usage:
# python eval_fasterrcnn_metrics.py --sensor evk4 --scene 0 --model_path /path/to/model.pth --input_dir /path/to/data --output_dir /path/to/output --batch_size 8 --gpu 0 --seed 42
"""
