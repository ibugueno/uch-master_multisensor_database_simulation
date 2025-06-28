import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm  # Importar tqdm para las barras de progreso


# -------------------------------------
# Dataset
# -------------------------------------
class PoseDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.txt')])
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Leer imagen
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')  # Garantiza 3 canales

        # Leer etiqueta
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        with open(label_path, 'r') as f:
            label = f.readline().strip().split(',')

        class_id = int(label[0])
        _, _, width, height, z_distance = map(float, label[1:6])
        q1, q2, q3, q4 = map(float, label[6:10])

        # Bounding Box
        x_min = max(0, int((float(label[1]) - width / 2) * image.size[0]))
        x_max = min(image.size[0], int((float(label[1]) + width / 2) * image.size[0]))
        y_min = max(0, int((float(label[2]) - height / 2) * image.size[1]))
        y_max = min(image.size[1], int((float(label[2]) + height / 2) * image.size[1]))

        # Recortar y redimensionar la imagen
        image_cropped = image.crop((x_min, y_min, x_max, y_max))
        image_cropped = image_cropped.resize((224, 224))  # Redimensionar a 224x224
        image_cropped = self.transforms(image_cropped)  # Convertir a tensor [3, 224, 224]

        # Pose (z, q1, q2, q3, q4)
        pose = torch.tensor([z_distance, q1, q2, q3, q4], dtype=torch.float32)

        return image_cropped, pose, class_id




# -------------------------------------
# Modelo PoseRegressor
# -------------------------------------
class PoseRegressor(nn.Module):
    def __init__(self, num_classes=20):
        super(PoseRegressor, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Reduce a [batch_size, 512, 1, 1]

        # Reemplazar la capa fc de ResNet para que acepte [batch_size, 512]
        self.cnn.fc = nn.Linear(512, 5 + num_classes)  # Salida para pose (5) + clasificación (num_classes)

    def forward(self, image):
        x = self.cnn.conv1(image)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.cnn.layer4(x)

        x = self.cnn.avgpool(x)  # Reduce a [batch_size, 512, 1, 1]
        x = torch.flatten(x, 1)  # Convierte a [batch_size, 512]
        x = self.cnn.fc(x)  # Salida final [batch_size, 5 + num_classes]

        pose = x[:, :5]           # Primeros 5 valores para pose
        class_logits = x[:, 5:]   # Resto para clasificación
        return pose, class_logits




# -------------------------------------
# Entrenamiento
# -------------------------------------
def train_model(model, train_loader, val_loader, optimizer, pose_criterion, epochs, device, save_path):
    for epoch in range(epochs):
        model.train()
        train_pose_loss = 0.0

        # Barra de progreso para entrenamiento
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", unit="batch")

        for images, poses, _ in train_loader_tqdm:
            images, poses = images.to(device), poses.to(device)

            optimizer.zero_grad()
            pred_pose, _ = model(images)
            pose_loss = pose_criterion(pred_pose, poses)
            pose_loss.backward()
            optimizer.step()
            train_pose_loss += pose_loss.item()

            # Actualizar descripción de la barra
            train_loader_tqdm.set_postfix({"Train Loss": f"{train_pose_loss / len(train_loader):.4f}"})

        # Validación al final de cada época
        val_pose_loss = 0.0
        model.eval()

        # Barra de progreso para validación
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation", unit="batch")

        with torch.no_grad():
            for images, poses, _ in val_loader_tqdm:
                images, poses = images.to(device), poses.to(device)

                pred_pose, _ = model(images)
                pose_loss = pose_criterion(pred_pose, poses)
                val_pose_loss += pose_loss.item()

                # Actualizar descripción de la barra
                val_loader_tqdm.set_postfix({"Val Loss": f"{val_pose_loss / len(val_loader):.4f}"})

        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"    Train Pose Loss: {train_pose_loss / len(train_loader):.4f}")
        print(f"    Val Pose Loss:   {val_pose_loss / len(val_loader):.4f}\n")

        # Guardar modelo al final de cada época
        torch.save(model.state_dict(), os.path.join(save_path, f"pose_model_epoch_{epoch+1}.pth"))




# -------------------------------------
# Main
# -------------------------------------
if __name__ == "__main__":


    # Obtener el nombre del archivo actual y extraer información
    current_file = os.path.basename(__file__)
    exp = int(current_file[:1])
    sensor = 'asus'


    # Leer configuración YAML
    with open(f"data/{sensor}_{exp}_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Extraer paths y parámetros
    train_image_dir = config['train']['images']
    train_label_dir = config['train']['labels']

    val_image_dir = config['val']['images']
    val_label_dir = config['val']['labels']

    hyperparameters = config['hyperparameters']
    batch_size = hyperparameters['batch_size']
    learning_rate = hyperparameters['learning_rate']
    epochs = hyperparameters['epochs']
    device = hyperparameters['device']
    num_workers = hyperparameters['num_workers']
    num_classes = config['num_classes']

    save_path = config['output_dir']

    # Crear datasets y loaders
    train_dataset = PoseDataset(train_image_dir, train_label_dir)
    val_dataset = PoseDataset(val_image_dir, val_label_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Crear modelo, pérdida y optimizador
    model = PoseRegressor(num_classes=num_classes).to(device)
    pose_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Crear directorio para guardar modelos
    os.makedirs(save_path, exist_ok=True)

    # Entrenar modelo
    train_model(model, train_loader, val_loader, optimizer, pose_criterion, epochs, device, save_path)
