"""
train_pose_estimation.py (final full script with parse_args and clean main)
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import csv

def load_paths(data_txt, label_txt):
    with open(data_txt) as f:
        image_paths = [line.strip() for line in f]
    with open(label_txt) as f:
        label_paths = [line.strip() for line in f]
    return image_paths, label_paths

class PoseDataset(Dataset):
    def __init__(self, input_dir, sensor, scene):
        data_txt = os.path.join(input_dir, f"{sensor}_data_scene_{scene}.txt")
        label_txt = os.path.join(input_dir, f"{sensor}_pose6d-abs_scene_{scene}.txt")
        self.image_paths, self.label_paths = load_paths(data_txt, label_txt)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transforms(img)
        with open(self.label_paths[idx]) as f:
            lines = f.readlines()[1]
            data = [float(x) for x in lines.strip().split(',')[2:]]
            z = torch.tensor([data[0]], dtype=torch.float32)
            quat = torch.tensor(data[1:], dtype=torch.float32)
        return img, z, quat, os.path.basename(self.image_paths[idx])

class PoseResNet18(nn.Module):
    def __init__(self):
        super(PoseResNet18, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 5)

    def forward(self, x):
        return self.backbone(x)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_z_loss, total_q_loss = 0, 0, 0
    for images, z, quat, _ in tqdm(loader, desc='Train'):
        images, z, quat = images.to(device), z.to(device), quat.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        z_pred, quat_pred = outputs[:,0], outputs[:,1:]
        z_loss = criterion(z_pred, z.squeeze())
        q_loss = criterion(quat_pred, quat)
        loss = z_loss + q_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_z_loss += z_loss.item()
        total_q_loss += q_loss.item()
    n = len(loader)
    return total_loss/n, total_z_loss/n, total_q_loss/n

def validate(model, loader, criterion, device, log_writer):
    model.eval()
    total_loss, total_z_loss, total_q_loss = 0, 0, 0
    with torch.no_grad():
        for images, z, quat, img_names in tqdm(loader, desc='Val'):
            images, z, quat = images.to(device), z.to(device), quat.to(device)
            outputs = model(images)
            z_pred, quat_pred = outputs[:,0], outputs[:,1:]
            z_loss = criterion(z_pred, z.squeeze())
            q_loss = criterion(quat_pred, quat)
            loss = z_loss + q_loss
            total_loss += loss.item()
            total_z_loss += z_loss.item()
            total_q_loss += q_loss.item()
            for i in range(len(img_names)):
                log_writer.writerow({'image': img_names[i], 'z_loss': z_loss.item(), 'q_loss': q_loss.item(), 'total_loss': loss.item()})
    n = len(loader)
    return total_loss/n, total_z_loss/n, total_q_loss/n

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = PoseDataset(args.input_dir, args.sensor, args.scene)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = PoseResNet18().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    csv_log_path = os.path.join(args.output_dir, 'epoch_log.csv')
    with open(csv_log_path, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'phase', 'total_loss', 'z_loss', 'q_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(args.epochs):
            tl, zl, ql = train_one_epoch(model, loader, optimizer, criterion, args.device)
            writer.writerow({'epoch': epoch+1, 'phase': 'train', 'total_loss': tl, 'z_loss': zl, 'q_loss': ql})

            val_log_path = os.path.join(args.output_dir, f'val_epoch{epoch+1}.csv')
            with open(val_log_path, 'w', newline='') as valcsv:
                val_writer = csv.DictWriter(valcsv, fieldnames=['image', 'z_loss', 'q_loss', 'total_loss'])
                val_writer.writeheader()
                vl, vzl, vql = validate(model, loader, criterion, args.device, val_writer)
            writer.writerow({'epoch': epoch+1, 'phase': 'val', 'total_loss': vl, 'z_loss': vzl, 'q_loss': vql})

            torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_epoch{epoch+1}.pth'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--sensor', required=True)
    parser.add_argument('--scene', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)

"""
✅ Script completo y ordenado con parse_args, main(args), dataset modular, entrenamiento y validación listos.
