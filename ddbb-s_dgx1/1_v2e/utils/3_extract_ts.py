import os
from dv import AedatFile
import numpy as np
import cv2
import time
from tqdm import tqdm

class TimestampImage:
    def __init__(self, sensor_size):
        self.sensor_size = sensor_size
        self.image = np.full(sensor_size, -np.inf, dtype=np.float64)

    def add_event(self, x, y, t):
        if t > self.image[int(y), int(x)]:
            self.image[int(y), int(x)] = t

    def add_events(self, xs, ys, ts):
        for x, y, t in zip(xs, ys, ts):
            self.add_event(x, y, t)

    def clear_old_events(self, current_time, window_size_us):
        self.image[self.image < current_time - window_size_us] = -np.inf

    def get_image(self, current_time, window_size_us):
        self.clear_old_events(current_time, window_size_us)
        valid_mask = self.image > -np.inf
        min_value = np.min(self.image[valid_mask]) if np.any(valid_mask) else 0
        max_value = np.max(self.image[valid_mask]) if np.any(valid_mask) else 1
        normalized_image = (self.image - min_value) / (max_value - min_value + 1e-9)
        normalized_image[~valid_mask] = 0
        return normalized_image


def generate_frames_online(aedat_file_path, sensor_size, output_folder, start_index, window_size_us=15_000, step_us=1_000):
    os.makedirs(output_folder, exist_ok=True)
    frame_index = start_index
    current_time = 0
    ts_img = TimestampImage(sensor_size)

    with AedatFile(aedat_file_path) as f:
        if 'events' not in f.names:
            print(f"No se encontró el stream de eventos en el archivo: {aedat_file_path}")
            return

        progress_bar = tqdm(desc="Procesando frames", unit="frame")
        last_event_time = 0

        for event in f['events']:
            x, y, t = event.x, event.y, event.timestamp
            last_event_time = t
            ts_img.add_event(x, y, t)

            if t >= current_time + window_size_us:
                timestamp_image = ts_img.get_image(current_time, window_size_us)
                normalized_image = (timestamp_image * 255).astype(np.uint8)
                colored_frame = cv2.applyColorMap(normalized_image, cv2.COLORMAP_VIRIDIS)

                frame_path = os.path.join(output_folder, f"image_{frame_index:04d}.jpg")
                cv2.imwrite(frame_path, colored_frame)
                frame_index += 1
                current_time += step_us
                progress_bar.update(1)

        while current_time <= last_event_time:
            timestamp_image = ts_img.get_image(current_time, window_size_us)
            normalized_image = (timestamp_image * 255).astype(np.uint8)
            colored_frame = cv2.applyColorMap(normalized_image, cv2.COLORMAP_VIRIDIS)
            frame_path = os.path.join(output_folder, f"image_{frame_index:04d}.jpg")
            cv2.imwrite(frame_path, colored_frame)
            frame_index += 1
            current_time += step_us
            progress_bar.update(1)

        progress_bar.close()

    print(f"Frames guardados en: {output_folder}")


def get_start_index(reference_folder):
    if not os.path.exists(reference_folder):
        print(f"La carpeta {reference_folder} no existe.")
        return 0

    image_files = [f for f in os.listdir(reference_folder) if f.startswith("image_") and f.endswith(".jpg")]
    if not image_files:
        print(f"No se encontraron archivos en {reference_folder}.")
        return 0

    indices = [int(f.split('_')[1].split('.')[0]) for f in image_files]
    return min(indices)


def process_aedat_files_online(txt_file, aedat_prefix, image_index_prefix, output_base_dir, sensor_size):


    with open(txt_file, 'r') as file:
        aedat_paths = [line.strip() for line in file.readlines()]

    for relative_path in aedat_paths:


        aedat_file_path = os.path.join(aedat_prefix, relative_path)
        reference_folder = os.path.join(image_index_prefix, os.path.dirname(relative_path))
        output_folder = os.path.join(output_base_dir, os.path.dirname(relative_path))

        start_index = get_start_index(reference_folder)

        print(f"Procesando: {aedat_file_path}")
        print(f"Referencia: {reference_folder} (Comenzando desde índice {start_index:04d})")
        print(f"Guardando en: {output_folder}")

        generate_frames_online(aedat_file_path, sensor_size, output_folder, start_index)


# Configuración
sensor_size = (720, 1280)
txt_file = "../data/files_list.txt"  # Archivo con rutas relativas a .aedat4
aedat_prefix = "../data/events"
image_index_prefix = "../data/frames"
output_base_dir = "../data/events_image"

# Inicio de la medición
start_time = time.time()

process_aedat_files_online(txt_file, aedat_prefix, image_index_prefix, output_base_dir, sensor_size)

print(f"Tiempo transcurrido metodo online: {time.time() - start_time:.4f} segundos")
