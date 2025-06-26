import os
from dv import AedatFile
import numpy as np
import cv2
import time

class TimestampImage:
    def __init__(self, sensor_size):
        self.sensor_size = sensor_size
        self.image = np.ones(sensor_size)

    def set_init(self, value):
        self.image = np.ones_like(self.image) * value

    def add_event(self, x, y, t):
        self.image[int(y), int(x)] = t

    def add_events(self, xs, ys, ts):
        for x, y, t in zip(xs, ys, ts):
            self.add_event(x, y, t)

    def get_image(self):
        normalized_image = (self.image - np.min(self.image)) / (np.max(self.image) - np.min(self.image))
        return normalized_image


def generate_frames_online(aedat_file_path, sensor_size, output_folder, start_index, window_size_us=5_000, step_us=1_000):
    os.makedirs(output_folder, exist_ok=True)

    with AedatFile(aedat_file_path) as f:
        if 'events' not in f.names:
            print(f"No se encontró el stream de eventos en el archivo: {aedat_file_path}")
            return

        frame_index = start_index
        current_time = 0
        buffer_events = {"x": [], "y": [], "timestamp": []}

        for event in f['events']:
            # Agregar evento al buffer
            buffer_events["x"].append(event.x)
            buffer_events["y"].append(event.y)
            buffer_events["timestamp"].append(event.timestamp)

            # Si alcanzamos el final de la ventana actual, generamos el frame
            if event.timestamp >= current_time + window_size_us:
                ts_img = TimestampImage(sensor_size)

                # Filtrar eventos dentro de la ventana de tiempo actual
                window_events = {
                    "x": [],
                    "y": [],
                    "timestamp": [],
                }

                for x, y, t in zip(buffer_events["x"], buffer_events["y"], buffer_events["timestamp"]):
                    if current_time <= t < current_time + window_size_us:
                        window_events["x"].append(x)
                        window_events["y"].append(y)
                        window_events["timestamp"].append(t)

                # Generar la imagen para el frame actual
                if window_events["timestamp"]:
                    ts_img.set_init(window_events["timestamp"][0])
                    ts_img.add_events(window_events["x"], window_events["y"], window_events["timestamp"])
                    timestamp_image = ts_img.get_image()
                else:
                    timestamp_image = np.zeros(sensor_size)

                normalized_image = (timestamp_image * 255).astype(np.uint8)
                colored_frame = cv2.applyColorMap(normalized_image, cv2.COLORMAP_VIRIDIS)

                frame_path = os.path.join(output_folder, f"image_{frame_index:04d}.jpg")
                cv2.imwrite(frame_path, colored_frame)

                print(f"Frame {frame_index} generado hasta {current_time + window_size_us} µs")
                current_time += step_us
                frame_index += 1

                # Limpiar eventos antiguos en el buffer
                buffer_events = {
                    "x": [x for x, t in zip(buffer_events["x"], buffer_events["timestamp"]) if t >= current_time],
                    "y": [y for y, t in zip(buffer_events["y"], buffer_events["timestamp"]) if t >= current_time],
                    "timestamp": [t for t in buffer_events["timestamp"] if t >= current_time],
                }

    print(f"Frames guardados en: {output_folder}")


def get_start_index(reference_folder):
    """Obtiene el índice inicial de las imágenes en la carpeta de referencia."""
    #print(f"Revisando carpeta de referencia: {reference_folder}")
    if not os.path.exists(reference_folder):
        print(f"La carpeta {reference_folder} no existe.")
        return 0
    
    image_files = [f for f in os.listdir(reference_folder) if f.startswith("image_") and f.endswith(".jpg")]
    #print(f"Archivos encontrados: {image_files}")

    if not image_files:
        print(f"No se encontraron archivos en {reference_folder}.")
        return 0
    
    indices = [int(f.split('_')[1].split('.')[0]) for f in image_files]
    #print(f"Índice detectado: {min(indices)}")
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

print(f"Tiempo transcurrido metodo online: {time.time()-start_time:.4f} segundos")
