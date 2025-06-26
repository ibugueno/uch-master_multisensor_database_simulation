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


def generate_video_with_5ms_window(events, sensor_size, frame_interval_us, window_size_us, output_video_path):
    frame_width, frame_height = sensor_size[1], sensor_size[0]

    # Configuración del video
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Configuración inicial
    current_time = 0
    max_time = events[-1]["timestamp"]  # Último timestamp en los eventos

    while current_time + window_size_us <= max_time:
        next_time = current_time + frame_interval_us
        window_end_time = current_time + window_size_us

        frame_data = {"x": [], "y": [], "timestamp": []}

        # Filtrar eventos dentro de la ventana de tiempo
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

        # Normalizar y convertir a RGB para guardar en video
        normalized_image = (timestamp_image * 255).astype(np.uint8)
        colored_frame = cv2.applyColorMap(normalized_image, cv2.COLORMAP_VIRIDIS)

        # Escribir el frame en el video
        video_writer.write(colored_frame)

        # Avanzar al siguiente intervalo de tiempo
        current_time = next_time

    video_writer.release()
    print(f"Video guardado en: {output_video_path}")


# Dimensiones del sensor (debe configurarse según el sensor utilizado)
sensor_size = (260, 346)

# Ruta al archivo AEDAT4
aedat_file_path = "output/almohada/events.aedat4"

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

# Generar video de TimestampImage con ventanas de 5ms y pasos de 1ms
frame_interval_us = 1_000  # Intervalo entre frames en microsegundos (1ms)
window_size_us = 5_000  # 5_000, lanzamiento Tamaño de la ventana de tiempo en microsegundos (5ms)
output_video_path = "output/almohada/output_timestamp_image_aedat4.avi"

generate_video_with_5ms_window(events, sensor_size, frame_interval_us, window_size_us, output_video_path)
