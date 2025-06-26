import os
from dv import AedatFile
import numpy as np
import cv2


# Definiciones de TimestampImage y Visualizador
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


def generate_frames_every_1ms(events, sensor_size, output_folder, window_size_us=5_000):
    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Obtener el timestamp máximo en los eventos
    max_time = events[-1]["timestamp"] if events else 0

    frame_index = 0  # Contador de frames
    current_time = 0  # Tiempo inicial en microsegundos

    while current_time <= max_time:
        window_end_time = current_time + window_size_us  # Fin de la ventana de 5 ms

        # Filtrar eventos dentro de la ventana de tiempo
        frame_data = {"x": [], "y": [], "timestamp": []}
        for e in events:
            if current_time <= e["timestamp"] < window_end_time:
                frame_data["x"].append(e["x"])
                frame_data["y"].append(e["y"])
                frame_data["timestamp"].append(e["timestamp"])

        # Crear imagen de TimestampImage
        ts_img = TimestampImage(sensor_size)
        if frame_data["timestamp"]:
            ts_img.set_init(frame_data["timestamp"][0])
            ts_img.add_events(frame_data["x"], frame_data["y"], frame_data["timestamp"])
            timestamp_image = ts_img.get_image()
        else:
            timestamp_image = np.zeros(sensor_size)  # Imagen vacía si no hay eventos

        # Normalizar y convertir a RGB
        normalized_image = (timestamp_image * 255).astype(np.uint8)
        colored_frame = cv2.applyColorMap(normalized_image, cv2.COLORMAP_VIRIDIS)

        # Guardar el frame como imagen .jpg
        frame_path = os.path.join(output_folder, f"image_{frame_index:04d}.jpg")
        cv2.imwrite(frame_path, colored_frame)

        # Avanzar al siguiente intervalo de tiempo (1 ms)
        current_time += 1_000
        frame_index += 1

    print(f"Frames guardados en: {output_folder}")


# Dimensiones del sensor (debe configurarse según el sensor utilizado)
sensor_size = (260, 346)

# Ruta al archivo AEDAT4
aedat_file_path = "../output/almohada_4_events_noisy/events.aedat4"

# Leer todos los eventos del archivo AEDAT4
events = []
with AedatFile(aedat_file_path) as f:
    if 'events' in f.names:
        for event in f['events']:
            events.append(
                {
                    "x": event.x,
                    "y": event.y,
                    "timestamp": event.timestamp,
                }
            )
    else:
        print("No se encontró el stream de eventos en el archivo.")

# Generar frames de TimestampImage cada 1ms con ventana de 5ms
output_folder = "../tmp/almohada_4_events_noisy/"

generate_frames_every_1ms(events, sensor_size, output_folder, window_size_us=5_000)
