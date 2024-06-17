# Lista de clases de objetos
'''

'''

import bpy
import math
import os
import shutil
from scipy.spatial.transform import Rotation as R

# Configurar Blender para usar la GPU (Metal en macOS)

bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.preferences.addons['cycles'].preferences.get_devices()
for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    
    if (device.name == 'Tesla V100-SXM2-32GB'):
        print('device: ', device)
        device.use = True
        print(f"Usando dispositivo GPU: {device.name}")

# Lista de clases de objetos
object_classes = [
    'almohada', 'arbol', 'avion', 'boomerang', 'caja_amarilla', 'caja_azul', 
    'carro_rojo', 'clorox', 'dino', 'disco', 'jarron', 'lysoform', 'mobil', 
    'paleta', 'pelota', 'sombrero', 'tarro', 'tazon', 'toalla_roja', 'zapatilla'
]

# Ruta base para los archivos .obj y las carpetas de salida
base_path = "../input/all/" 
output_base_folder_path = "../output/output_frames/"  # Carpeta base donde se guardarán las imágenes

# Eliminar la carpeta de salida si existe y crearla nuevamente
if os.path.exists(output_base_folder_path):
    shutil.rmtree(output_base_folder_path)
os.makedirs(output_base_folder_path)

# Parámetros del movimiento parabólico
num_frames = 1500  # Número de frames para la animación
start_x = -2  # Posición inicial en X
end_x = 2  # Posición final en X
height_factor = 1.5  # Factor de altura máxima de la parábola

# Generar 24 orientaciones únicas utilizando quaternions
num_orientations = 6 # 24
rotations = R.random(num_orientations)
orientations_degrees = rotations.as_euler('xyz', degrees=True)

# Generar y guardar imágenes con desplazamiento parabólico para cada clase y orientación
for object_class in object_classes:
    obj_path = os.path.join(base_path, f"{object_class}/{object_class}.obj")
    class_output_folder_path = os.path.join(output_base_folder_path, object_class)
    
    # Crear las carpetas de salida para la clase si no existen
    if not os.path.exists(class_output_folder_path):
        os.makedirs(class_output_folder_path)
    
    # Limpiar la escena actual
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Crear una luz tipo 'SUN'
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_object = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_object)

    # Posicionar la luz en la escena
    light_object.location = (10, -10, 10)

    # Ajustar la dirección de la luz (rotación)
    light_object.rotation_euler[0] = math.radians(45)
    light_object.rotation_euler[1] = math.radians(0)
    light_object.rotation_euler[2] = math.radians(45)

    # Ajustar la energía y otros atributos de la luz
    light_data.energy = 5

    # Importar el archivo .obj
    bpy.ops.import_scene.obj(filepath=obj_path)

    # Obtener la referencia del objeto importado
    imported_obj = bpy.context.selected_objects[0]

    # Calcular el centro y dimensiones del objeto
    obj_center = imported_obj.location
    obj_dimensions = imported_obj.dimensions
    max_dimension = max(obj_dimensions)

    # Crear una nueva cámara
    camera_data = bpy.data.cameras.new(name='Camera')
    camera_object = bpy.data.objects.new('Camera', camera_data)
    bpy.context.scene.collection.objects.link(camera_object)

    # Posicionar la cámara centrada en el objeto y a una distancia tal que el objeto quepa en el marco
    camera_object.location = (0, -max_dimension * 4, max_dimension * height_factor)
    camera_object.rotation_euler = (math.pi / 2, 0, 0)

    # Configurar la escena para usar la cámara recién creada
    bpy.context.scene.camera = camera_object

    # Configurar la renderización
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.context.scene.render.resolution_x = 1280 # 346
    bpy.context.scene.render.resolution_y = 720 #260
    bpy.context.scene.render.resolution_percentage = 100

    # Configurar la cámara
    camera_data.lens = 50
    camera_data.sensor_width = 36

    for orientation_deg in orientations_degrees:
        # Configurar la rotación inicial del objeto
        orientation_rad = [math.radians(angle) for angle in orientation_deg]
        imported_obj.rotation_euler = orientation_rad
        
        # Crear carpeta para la orientación específica
        orientation_folder_path = os.path.join(class_output_folder_path, f'{object_class}_orientation_{int(orientation_deg[0])}_{int(orientation_deg[1])}_{int(orientation_deg[2])}')
        if not os.path.exists(orientation_folder_path):
            os.makedirs(orientation_folder_path)
        
        for i in range(num_frames):
            t = i / (num_frames - 1)
            x = start_x + t * (end_x - start_x)
            y = -4 * max_dimension * height_factor * (t - 0.5) ** 2 + max_dimension * height_factor  # Ecuación de la parábola (forma estándar)
            
            # Posicionar el objeto en la nueva posición
            imported_obj.location.x = x
            imported_obj.location.z = y  # Ajustar la altura para el movimiento parabólico

            # Aplicar una rotación sobre su propio eje mientras se mueve
            rotation_angle = t * 2 * math.pi / 540  # Rotación completa (360 grados) durante el movimiento
            imported_obj.rotation_euler[2] += rotation_angle  # Ajustar la rotación en el eje Z (puedes cambiar el eje según sea necesario)

            # Configurar el nombre del archivo de salida
            output_image_path = os.path.join(orientation_folder_path, f'image_{i:04d}.jpg')
            
            # Configurar la ruta del archivo de renderizado
            bpy.context.scene.render.filepath = output_image_path
            
            # Renderizar la imagen
            bpy.ops.render.render(write_still=True)

        break

    break

print("Generación de imágenes completada.")
