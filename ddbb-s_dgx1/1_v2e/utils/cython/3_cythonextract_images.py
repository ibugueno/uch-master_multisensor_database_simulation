from cython_event_processing import process_aedat_files_online
import time

# Configuración general
txt_file = "../data/files_list.txt"
aedat_prefix = "../data/events"
image_index_prefix = "../data/frames"
output_base_dir = "../data/events_image"

# Inicio de la medición
start_time = time.time()

process_aedat_files_online(txt_file, aedat_prefix, image_index_prefix, output_base_dir)

print(f"Tiempo transcurrido metodo online: {time.time() - start_time:.4f} segundos")
