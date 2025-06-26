import os
from dv import AedatFile
import numpy as np
import cv2
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Configuración
height = 720
width = 1280
time_window_us = 5_000  # Ventana de tiempo en microsegundos
step_us = 1_000  # Paso entre ventanas de tiempo
background_color = np.array([52, 37, 30], dtype=np.uint8)  # Color de fondo en BGR
positive_event_color = np.array([255, 255, 255], dtype=np.uint8)  # Color para eventos positivos en BGR
negative_event_color = np.array([201, 126, 64], dtype=np.uint8)  # Color para eventos negativos en BGR


def generate_frames_online(aedat_file_path, output_folder, start_index):
    """
    Genera frames acumulando eventos dentro de cada ventana de tiempo.
    """
    os.makedirs(output_folder, exist_ok=True)
    frame_index = start_index
    current_time = 0

    # Inicializa la imagen de eventos
    event_image = np.full((height, width, 3), background_color, dtype=np.uint8)

    with AedatFile(aedat_file_path) as f:
        if 'events' not in f.names:
            print(f"No se encontró el stream de eventos en el archivo: {aedat_file_path}")
            return

        # Obtén el timestamp del último evento para calcular el progreso
        last_event_timestamp = max(event.timestamp for event in f['events'])
        total_frames = int((last_event_timestamp - current_time) / step_us)
        progress_bar = tqdm(total=total_frames, desc="Frames procesados", unit="frame")

        # Iterar sobre eventos
        for event in f['events']:
            x, y, t, polarity = event.x, event.y, event.timestamp, event.polarity

            if t >= current_time + time_window_us:
                # Guardar el frame actual
                frame_path = os.path.join(output_folder, f"image_{frame_index:04d}.jpg")
                cv2.imwrite(frame_path, event_image)

                # Incrementar tiempo y frame
                frame_index += 1
                current_time += step_us
                progress_bar.update(1)

                # Reiniciar solo si se salió de la ventana de tiempo
                if t >= current_time + time_window_us:
                    event_image[:, :] = background_color

            # Asignar el evento al frame actual
            color = positive_event_color if polarity else negative_event_color
            event_image[y, x] = color

        # Guardar el último frame si hay eventos residuales
        frame_path = os.path.join(output_folder, f"image_{frame_index:04d}.jpg")
        cv2.imwrite(frame_path, event_image)
        progress_bar.update(1)

        progress_bar.close()

    print(f"Frames guardados en: {output_folder}")


def get_start_index(reference_folder):
    """Obtiene el índice inicial de las imágenes en la carpeta de referencia."""
    if not os.path.exists(reference_folder):
        print(f"La carpeta {reference_folder} no existe.")
        return 0

    image_files = [f for f in os.listdir(reference_folder) if f.startswith("image_") and f.endswith(".jpg")]
    if not image_files:
        print(f"No se encontraron archivos en {reference_folder}.")
        return 0

    indices = [int(f.split('_')[1].split('.')[0]) for f in image_files]
    return min(indices)


def process_file(aedat_file_path, reference_folder, output_folder):
    """Procesa un único archivo AEDAT."""
    start_index = get_start_index(reference_folder)
    print(f"Procesando: {aedat_file_path}")
    print(f"Referencia: {reference_folder} (Comenzando desde índice {start_index:04d})")
    print(f"Guardando en: {output_folder}")
    generate_frames_online(aedat_file_path, output_folder, start_index)


def process_aedat_files_online_parallel(txt_file, aedat_prefix, image_index_prefix, output_base_dir):
    """
    Procesa múltiples archivos AEDAT en paralelo.
    """
    with open(txt_file, 'r') as file:
        aedat_paths = [line.strip() for line in file.readlines()]

    tasks = []
    with ThreadPoolExecutor() as executor:
        for relative_path in aedat_paths:
            aedat_file_path = os.path.join(aedat_prefix, relative_path)
            reference_folder = os.path.join(image_index_prefix, os.path.dirname(relative_path))
            output_folder = os.path.join(output_base_dir, os.path.dirname(relative_path))

            # Enviar tarea al executor
            tasks.append(executor.submit(process_file, aedat_file_path, reference_folder, output_folder))

        # Asegurarse de que todas las tareas hayan terminado
        for task in tqdm(tasks, desc="Procesando archivos", unit="archivo"):
            task.result()


# Configuración general
txt_file = "../data/files_list.txt"  # Archivo con rutas relativas a .aedat4
aedat_prefix = "../data/events"
image_index_prefix = "../data/frames"
output_base_dir = "../data/events_image"

# Inicio de la medición
start_time = time.time()

process_aedat_files_online_parallel(txt_file, aedat_prefix, image_index_prefix, output_base_dir)

print(f"Tiempo transcurrido metodo online: {time.time() - start_time:.4f} segundos")
