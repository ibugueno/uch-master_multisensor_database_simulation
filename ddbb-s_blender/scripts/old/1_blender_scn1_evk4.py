import bpy
import math
import os
import shutil
import yaml
import numpy as np
from datetime import datetime
from mathutils import Vector

# Leer configuraciones de cámara desde un archivo YAML
yaml_file = "../sensors/config_evk4.yaml"
with open(yaml_file, "r") as file:
    camera_config = yaml.safe_load(file)

# Extraer valores del YAML
h_fov = camera_config['camera']['H-FOV']
v_fov = camera_config['camera']['V-FOV']

# Calcular tamaño del sensor basado en FOV y lente
lens_mm = 50  # Valor fijo para la lente
sensor_width = 2 * lens_mm * math.tan(math.radians(h_fov) / 2)
sensor_height = 2 * lens_mm * math.tan(math.radians(v_fov) / 2)

# Configuración inicial
orientations_file = "../objects/orientations_24.txt"
orientations_degrees = np.loadtxt(orientations_file, skiprows=1)

bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'METAL'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.preferences.addons['cycles'].preferences.get_devices()
for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    device.use = True

object_classes = [
    'arbol', 'avion', 'boomerang', 'caja_amarilla', 'caja_azul',
    'carro_rojo', 'clorox', 'dino', 'disco', 'jarron', 'lysoform', 'mobil',
    'paleta', 'pelota', 'sombrero', 'tarro', 'tazon', 'toalla_roja', 'zapatilla'
]

base_path = "../objects/all/"
current_datetime = datetime.now().strftime("%Y%m%d_%H%M")
output_base_folder_path = f"../data/evk4/{current_datetime}/esc_1_parabolico_sin_fondo/"

if os.path.exists(output_base_folder_path):
    shutil.rmtree(output_base_folder_path)
os.makedirs(output_base_folder_path)

num_frames = 1500
start_x = -1
end_x = 1
height_factor = 1.5

# Generar y guardar imágenes para cada objeto y orientación
for object_class in object_classes:
    obj_path = os.path.join(base_path, f"{object_class}/{object_class}.obj")
    object_folder_path = os.path.join(output_base_folder_path, object_class)
    if not os.path.exists(object_folder_path):
        os.makedirs(object_folder_path)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

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

    # Configurar la cámara con los valores del YAML
    camera_data.lens = lens_mm
    camera_data.sensor_width = sensor_width
    camera_data.sensor_height = sensor_height

    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720
    bpy.context.scene.render.resolution_percentage = 100

    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_object = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = (10, -10, 10)
    light_object.rotation_euler = (math.radians(45), 0, math.radians(45))

    for lum_index, luminance in enumerate([1, 3, 9]):
        light_data.energy = luminance
        lum_folder_path = os.path.join(object_folder_path, f'lum{luminance}')
        if not os.path.exists(lum_folder_path):
            os.makedirs(lum_folder_path)

        for orientation_deg in orientations_degrees:
            orientation_rad = [math.radians(angle) for angle in orientation_deg]
            imported_obj.rotation_euler = orientation_rad

            orientation_str = f"{int(orientation_deg[0])}_{int(orientation_deg[1])}_{int(orientation_deg[2])}"
            orientation_folder_path = os.path.join(lum_folder_path, f'orientation_{orientation_str}')
            if not os.path.exists(orientation_folder_path):
                os.makedirs(orientation_folder_path)

            # Crear archivo para almacenar las poses 6D de esta orientación
            pose_file_path = os.path.join(orientation_folder_path, f"{object_class}_pose_orientation_{orientation_str}.txt")
            with open(pose_file_path, "w") as pose_file:
                pose_file.write("frame,x,y,z,qx,qy,qz,qw\n")  # Cabecera

                for i in range(num_frames):
                    t = i / (num_frames - 1)
                    x = start_x + t * (end_x - start_x)
                    y = -4 * max_dimension * height_factor * (t - 0.5) ** 2 + max_dimension * height_factor

                    imported_obj.location.x = x
                    imported_obj.location.z = y
                    imported_obj.rotation_euler[2] += t * 2 * math.pi / 540

                    # Actualizar la escena para reflejar los cambios
                    bpy.context.view_layer.update()

                    if 500 <= i <= 1000:
                        # Calcular posición relativa
                        relative_position = camera_object.matrix_world.inverted_safe() @ imported_obj.matrix_world.translation
                        relative_rotation = camera_object.matrix_world.inverted_safe().to_quaternion() @ imported_obj.matrix_world.to_quaternion()
                        z_distance = abs(relative_position.z)
                        print('z_distance:', z_distance)

                        if z_distance > 0:
                            x_pixel = (relative_position.x / z_distance) * (camera_data.lens / sensor_width) * bpy.context.scene.render.resolution_x + bpy.context.scene.render.resolution_x / 2
                            y_pixel = -(relative_position.y / z_distance) * (camera_data.lens / sensor_height) * bpy.context.scene.render.resolution_y + bpy.context.scene.render.resolution_y / 2

                            x_pixel = int(x_pixel)
                            y_pixel = int(y_pixel)

                            # Guardar pose
                            pose_file.write(f"{i},{x_pixel},{y_pixel},{z_distance},{relative_rotation.x},{relative_rotation.y},{relative_rotation.z},{relative_rotation.w}\n")

                            # Renderizar imagen RGB
                            output_image_path = os.path.join(orientation_folder_path, f'image_{i:04d}.jpg')
                            bpy.context.scene.render.filepath = output_image_path
                            bpy.ops.render.render(write_still=True)


            break
