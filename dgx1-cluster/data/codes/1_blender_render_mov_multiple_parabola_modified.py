import bpy
import math
import os
import shutil
from scipy.spatial.transform import Rotation as R

# Configurar Blender para usar Cycles y la GPU (CUDA)
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.preferences.addons['cycles'].preferences.get_devices()

# Seleccionar GPU
for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    if device.type == 'CUDA':
        device.use = True
        print(f"Usando dispositivo GPU: {device.name}")

# Verificar que la GPU está seleccionada en todas las escenas
for scene in bpy.data.scenes:
    scene.cycles.device = 'GPU'

# Ajustar configuraciones para acelerar la renderización
bpy.context.scene.cycles.samples = 64
bpy.context.scene.cycles.use_denoising = True
bpy.context.scene.cycles.use_adaptive_sampling = True
bpy.context.scene.cycles.max_bounces = 4
bpy.context.scene.cycles.diffuse_bounces = 2
bpy.context.scene.cycles.glossy_bounces = 2
bpy.context.scene.cycles.transmission_bounces = 2
bpy.context.scene.cycles.volume_bounces = 2
bpy.context.scene.render.resolution_percentage = 50

# Lista de clases de objetos
object_classes = [
    'almohada', 'arbol', 'avion', 'boomerang', 'caja_amarilla', 'caja_azul', 
    'carro_rojo', 'clorox', 'dino', 'disco', 'jarron', 'lysoform', 'mobil', 
    'paleta', 'pelota', 'sombrero', 'tarro', 'tazon', 'toalla_roja', 'zapatilla'
]

# Ruta base para los archivos .obj y las carpetas de salida
base_path = "../input/all/"
output_base_folder_path = "../output/output_frames/"

# Eliminar la carpeta de salida si existe y crearla nuevamente
if os.path.exists(output_base_folder_path):
    shutil.rmtree(output_base_folder_path)
os.makedirs(output_base_folder_path)

# Parámetros del movimiento parabólico
num_frames = 1500
start_x = -2
end_x = 2
height_factor = 1.5

# Generar 24 orientaciones únicas utilizando quaternions
num_orientations = 6
rotations = R.random(num_orientations)
orientations_degrees = rotations.as_euler('xyz', degrees=True)

# Generar y guardar imágenes con desplazamiento parabólico para cada clase y orientación
for object_class in object_classes:
    obj_path = os.path.join(base_path, f"{object_class}/{object_class}.obj")
    class_output_folder_path = os.path.join(output_base_folder_path, object_class)
    
    if not os.path.exists(class_output_folder_path):
        os.makedirs(class_output_folder_path)
    
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_object = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_object)

    light_object.location = (10, -10, 10)
    light_object.rotation_euler = (math.radians(45), math.radians(0), math.radians(45))
    light_data.energy = 5

    bpy.ops.import_scene.obj(filepath=obj_path)
    imported_obj = bpy.context.selected_objects[0]

    obj_center = imported_obj.location
    obj_dimensions = imported_obj.dimensions
    max_dimension = max(obj_dimensions)

    camera_data = bpy.data.cameras.new(name='Camera')
    camera_object = bpy.data.objects.new('Camera', camera_data)
    bpy.context.scene.collection.objects.link(camera_object)

    camera_object.location = (0, -max_dimension * 4, max_dimension * height_factor)
    camera_object.rotation_euler = (math.pi / 2, 0, 0)
    bpy.context.scene.camera = camera_object

    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720
    bpy.context.scene.render.resolution_percentage = 100

    camera_data.lens = 50
    camera_data.sensor_width = 36

    for orientation_deg in orientations_degrees:
        orientation_rad = [math.radians(angle) for angle in orientation_deg]
        imported_obj.rotation_euler = orientation_rad
        
        orientation_folder_path = os.path.join(class_output_folder_path, f'{object_class}_orientation_{int(orientation_deg[0])}_{int(orientation_deg[1])}_{int(orientation_deg[2])}')
        if not os.path.exists(orientation_folder_path):
            os.makedirs(orientation_folder_path)
        
        for i in range(num_frames):
            t = i / (num_frames - 1)
            x = start_x + t * (end_x - start_x)
            y = -4 * max_dimension * height_factor * (t - 0.5) ** 2 + max_dimension * height_factor
            
            imported_obj.location.x = x
            imported_obj.location.z = y

            rotation_angle = t * 2 * math.pi / 540
            imported_obj.rotation_euler[2] += rotation_angle

            output_image_path = os.path.join(orientation_folder_path, f'image_{i:04d}.jpg')
            bpy.context.scene.render.filepath = output_image_path
            bpy.ops.render.render(write_still=True)

        break

    break

print("Generación de imágenes completada.")
