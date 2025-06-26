# cython_event_processing.pyx

from dv import AedatFile
import os
import numpy as np
cimport numpy as cnp
import cv2

# Configuración global
cdef int height = 720
cdef int width = 1280
cdef int time_window_us = 5000
cdef int step_us = 1000

# Inicializa los colores de eventos como variables globales
background_color = np.array([52, 37, 30], dtype=np.uint8)
positive_event_color = np.array([255, 255, 255], dtype=np.uint8)
negative_event_color = np.array([201, 126, 64], dtype=np.uint8)


def generate_frames_online(str aedat_file_path, str output_folder, int start_index):
    """
    Genera frames acumulando eventos dentro de cada ventana de tiempo.
    """
    os.makedirs(output_folder, exist_ok=True)
    cdef int frame_index = start_index
    cdef int current_time = 0

    # Inicializa la imagen de eventos
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] event_image = np.full((height, width, 3), background_color, dtype=np.uint8)

    # Declarar buffers para almacenar lotes
    cdef list x_buffer = []
    cdef list y_buffer = []
    cdef list color_buffer = []

    cdef int x, y, t
    cdef bint polarity

    with AedatFile(aedat_file_path) as f:
        if 'events' not in f.names:
            print(f"No se encontró el stream de eventos en el archivo: {aedat_file_path}")
            return

        for event in f['events']:
            # Extraer datos del evento
            x = event.x
            y = event.y
            t = event.timestamp
            polarity = event.polarity

            # Agregar eventos al buffer
            x_buffer.append(x)
            y_buffer.append(y)
            color_buffer.append(positive_event_color if polarity else negative_event_color)

            # Si alcanzamos el final de la ventana de tiempo, procesar el frame
            if t >= current_time + time_window_us:
                if x_buffer:
                    # Asignar colores al lote
                    event_image[np.array(y_buffer), np.array(x_buffer)] = np.array(color_buffer)

                # Guardar el frame generado
                frame_path = os.path.join(output_folder, f"image_{frame_index:04d}.jpg")
                cv2.imwrite(frame_path, event_image)

                print(f"Frame {frame_index} generado hasta {current_time + time_window_us} µs")

                # Reiniciar buffers y preparar para la siguiente ventana
                event_image[:, :] = background_color
                x_buffer.clear()
                y_buffer.clear()
                color_buffer.clear()

                current_time += step_us
                frame_index += 1

    print(f"Frames guardados en: {output_folder}")


def get_start_index(str reference_folder):
    """
    Obtiene el índice inicial de las imágenes en la carpeta de referencia.
    """
    if not os.path.exists(reference_folder):
        print(f"La carpeta {reference_folder} no existe.")
        return 0

    cdef list image_files = [f for f in os.listdir(reference_folder) if f.startswith("image_") and f.endswith(".jpg")]
    if not image_files:
        print(f"No se encontraron archivos en {reference_folder}.")
        return 0

    cdef list indices = [int(f.split('_')[1].split('.')[0]) for f in image_files]
    return min(indices)


def process_aedat_files_online(str txt_file, str aedat_prefix, str image_index_prefix, str output_base_dir):
    """
    Procesa múltiples archivos AEDAT en base a una lista.
    """
    with open(txt_file, 'r') as file:
        aedat_paths = [line.strip() for line in file.readlines()]

    cdef str aedat_file_path
    cdef str reference_folder
    cdef str output_folder
    cdef int start_index

    for relative_path in aedat_paths:
        aedat_file_path = os.path.join(aedat_prefix, relative_path)
        reference_folder = os.path.join(image_index_prefix, os.path.dirname(relative_path))
        output_folder = os.path.join(output_base_dir, os.path.dirname(relative_path))

        start_index = get_start_index(reference_folder)

        print(f"Procesando: {aedat_file_path}")
        print(f"Referencia: {reference_folder} (Comenzando desde índice {start_index:04d})")
        print(f"Guardando en: {output_folder}")

        generate_frames_online(aedat_file_path, output_folder, start_index)
