import bpy
import math
import os
import shutil
import numpy as np
from datetime import datetime  # Importar módulo datetime

# Leer orientaciones desde el archivo
orientations_file = "../objects/orientations_24.txt"  # Cambiar a "orientations_8.txt" si es necesario
orientations_degrees = np.loadtxt(orientations_file, skiprows=1)  # Saltar la cabecera del archivo

# Configurar Blender para usar la GPU
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'METAL'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.preferences.addons['cycles'].preferences.get_devices()
for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    print('device: ', device)
    device.use = True
    print(f"Usando dispositivo GPU: {device.name}")

# Lista de clases de objetos
object_classes = [
    'arbol', 'avion', 'boomerang', 'caja_amarilla', 'caja_azul', 
    'carro_rojo', 'clorox', 'dino', 'disco', 'jarron', 'lysoform', 'mobil', 
    'paleta', 'pelota', 'sombrero', 'tarro', 'tazon', 'toalla_roja', 'zapatilla'
]

# Rutas base
base_path = "../objects/all/"

# Obtener la fecha y hora actual en formato YYYYMMDD_HHMM
current_datetime = datetime.now().strftime("%Y%m%d_%H%M")

# Crear la ruta de salida con la fecha y hora
output_base_folder_path = f"../data/{current_datetime}/esc_1_parabolico_sin_fondo/"

# Eliminar la carpeta de salida si existe
if os.path.exists(output_base_folder_path):
    shutil.rmtree(output_base_folder_path)
os.makedirs(output_base_folder_path)

# Parámetros del movimiento parabólico
num_frames = 1500
start_x = -1 #2
end_x = 1 #2
height_factor = 1.5

# Configuraciones de luces con posiciones y rotaciones predefinidas
light_configs = [
    {'position': (10, -10, 10), 'rotation': (math.radians(45), 0, math.radians(45))},
    {'position': (10, 0, 10), 'rotation': (math.radians(45), 0, 0)},  # Hacia el eje Y positivo
    {'position': (10, 10, 10), 'rotation': (math.radians(45), 0, math.radians(-45))}  # Hacia el eje Y negativo
]

print(enumerate(zip(object_classes, light_configs)))

# Generar y guardar imágenes para cada objeto y orientación
for object_class in object_classes:  # Primero iterar sobre los objetos
    obj_path = os.path.join(base_path, f"{object_class}/{object_class}.obj")

    # Crear un subdirectorio para cada objeto
    object_folder_path = os.path.join(output_base_folder_path, object_class)
    if not os.path.exists(object_folder_path):
        os.makedirs(object_folder_path)

    # Limpiar la escena actual
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Importar el archivo .obj
    bpy.ops.import_scene.obj(filepath=obj_path)
    imported_obj = bpy.context.selected_objects[0]
    max_dimension = max(imported_obj.dimensions)

    # Crear cámara
    camera_data = bpy.data.cameras.new(name='Camera')
    camera_object = bpy.data.objects.new('Camera', camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    camera_object.location = (0, -1, max_dimension * height_factor)
    camera_object.rotation_euler = (math.pi / 2, 0, 0)
    bpy.context.scene.camera = camera_object

    # Configurar renderización
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720
    bpy.context.scene.render.resolution_percentage = 100
    camera_data.lens = 50
    camera_data.sensor_width = 36

    # Ahora iterar sobre cada configuración de foco
    for light_index, light_config in enumerate(light_configs):
        # Crear un subdirectorio para cada foco dentro del objeto
        foco_folder_path = os.path.join(object_folder_path, f'foco{light_index}')
        if not os.path.exists(foco_folder_path):
            os.makedirs(foco_folder_path)

        # Limpiar luces existentes
        bpy.ops.object.select_all(action='SELECT')
        for obj in bpy.context.selected_objects:
            if obj.type == 'LIGHT':
                bpy.data.objects.remove(obj, do_unlink=True)

        # Crear luz con configuración específica
        light_data = bpy.data.lights.new(name=f"Sun_{light_config['position']}", type='SUN')
        light_object = bpy.data.objects.new(name=f"Sun_{light_config['position']}", object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        light_object.location = light_config['position']
        light_object.rotation_euler = light_config['rotation']
        light_data.energy = 5

        # Aplicar cada orientación
        for orientation_deg in orientations_degrees:
            orientation_rad = [math.radians(angle) for angle in orientation_deg]
            imported_obj.rotation_euler = orientation_rad

            # Crear carpeta para la orientación específica
            orientation_str = f"{int(orientation_deg[0])}_{int(orientation_deg[1])}_{int(orientation_deg[2])}"
            orientation_folder_path = os.path.join(foco_folder_path, f'orientation_{orientation_str}')
            if not os.path.exists(orientation_folder_path):
                os.makedirs(orientation_folder_path)

            # Renderizar frames
            for i in range(num_frames):
                t = i / (num_frames - 1)
                x = start_x + t * (end_x - start_x)
                y = -4 * max_dimension * height_factor * (t - 0.5) ** 2 + max_dimension * height_factor

                imported_obj.location.x = x
                imported_obj.location.z = y
                imported_obj.rotation_euler[2] += t * 2 * math.pi / 540  # Rotación sobre el eje Z

                # Configurar archivo de salida
                output_image_path = os.path.join(orientation_folder_path, f'image_{i:04d}.jpg')
                bpy.context.scene.render.filepath = output_image_path
                bpy.ops.render.render(write_still=True)

            break
