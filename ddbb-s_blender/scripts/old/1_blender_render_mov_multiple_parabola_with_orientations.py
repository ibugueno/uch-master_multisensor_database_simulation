import bpy
import math
import os
import shutil
import numpy as np

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
    'almohada', 'arbol', 'avion', 'boomerang', 'caja_amarilla', 'caja_azul', 
    'carro_rojo', 'clorox', 'dino', 'disco', 'jarron', 'lysoform', 'mobil', 
    'paleta', 'pelota', 'sombrero', 'tarro', 'tazon', 'toalla_roja', 'zapatilla'
]

# Rutas base
base_path = "../objects/all/"
output_base_folder_path = "../data/esc_1_parabolico_sin_fondo/"

# Eliminar la carpeta de salida si existe
if os.path.exists(output_base_folder_path):
    shutil.rmtree(output_base_folder_path)
os.makedirs(output_base_folder_path)

# Parámetros del movimiento parabólico
num_frames = 1500
start_x = -2
end_x = 2
height_factor = 1.5

# Generar y guardar imágenes para cada clase y orientación
for object_class in object_classes:
    obj_path = os.path.join(base_path, f"{object_class}/{object_class}.obj")
    class_output_folder_path = os.path.join(output_base_folder_path, object_class)
    
    if not os.path.exists(class_output_folder_path):
        os.makedirs(class_output_folder_path)

    # Limpiar la escena actual
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Crear luz tipo 'SUN'
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_object = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = (10, -10, 10)
    light_object.rotation_euler = (math.radians(45), 0, math.radians(45))
    light_data.energy = 5

    # Importar el archivo .obj
    bpy.ops.import_scene.obj(filepath=obj_path)
    imported_obj = bpy.context.selected_objects[0]
    max_dimension = max(imported_obj.dimensions)

    # Crear cámara
    camera_data = bpy.data.cameras.new(name='Camera')
    camera_object = bpy.data.objects.new('Camera', camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    camera_object.location = (0, -max_dimension * 4, max_dimension * height_factor)
    camera_object.rotation_euler = (math.pi / 2, 0, 0)
    bpy.context.scene.camera = camera_object

    # Configurar renderización
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720
    bpy.context.scene.render.resolution_percentage = 100
    camera_data.lens = 50
    camera_data.sensor_width = 36

    # Aplicar cada orientación
    for orientation_deg in orientations_degrees:
        orientation_rad = [math.radians(angle) for angle in orientation_deg]
        imported_obj.rotation_euler = orientation_rad

        # Crear carpeta para la orientación específica
        orientation_str = f"{int(orientation_deg[0])}_{int(orientation_deg[1])}_{int(orientation_deg[2])}"
        orientation_folder_path = os.path.join(class_output_folder_path, f'{object_class}_orientation_{orientation_str}')
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
