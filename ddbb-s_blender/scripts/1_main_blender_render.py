import argparse
import bpy
import math
import os
import shutil
import yaml
import numpy as np
from datetime import datetime
from mathutils import Vector


def parse_arguments():
    """Define y analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Renderizado y generación de poses 6D.")

    parser.add_argument("--sensor", type=str, choices=["evk4", "davis346", "rog", "zed2"],
                        required=True, help="Selecciona el sensor a usar (evk4, davis346, rog, zed2).")

    parser.add_argument("--scene", type=int, choices=[1, 2, 3], required=True,
                        help="Selecciona la escena a simular: 1 (parabólica), 2 (caída libre), 3 (lanzamiento hacia la cámara).")

    parser.add_argument("--luminosities", type=int, nargs='+', choices=[1, 3, 9], default=[1, 3, 9],
                        help="Especifica los niveles de luminosidad a simular (por defecto: 1, 3, 9).")

    parser.add_argument("--output_dir", type=str, default="../data/",
                        help="Directorio base para guardar los datos generados.")

    return parser.parse_args()

def load_camera_config(sensor_name):
    """Carga la configuración de la cámara según el sensor seleccionado."""
    yaml_file = f"../sensors/config_{sensor_name}.yaml"
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


def calculate_sensor_size(h_fov, v_fov, lens_mm):
    """Calcula el tamaño del sensor basado en el FOV y la longitud focal."""
    sensor_width = 2 * lens_mm * math.tan(math.radians(h_fov) / 2)
    sensor_height = 2 * lens_mm * math.tan(math.radians(v_fov) / 2)
    return sensor_width, sensor_height


def configure_render_settings():
    """Configura las preferencias de renderizado para GPU."""
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'METAL'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        device.use = True


def setup_camera(sensor_width, sensor_height, lens_mm, max_dimension, height_factor):
    """Crea y configura una cámara."""
    camera_data = bpy.data.cameras.new(name='Camera')
    camera_object = bpy.data.objects.new('Camera', camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    camera_object.location = (0, -1, max_dimension * height_factor)
    camera_object.rotation_euler = (math.pi / 2, 0, 0)
    bpy.context.scene.camera = camera_object

    camera_data.lens = lens_mm
    camera_data.sensor_width = sensor_width
    camera_data.sensor_height = sensor_height
    return camera_object


def create_light():
    """Crea una luz de tipo SUN."""
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_object = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = (10, -10, 10)
    light_object.rotation_euler = (math.radians(45), 0, math.radians(45))
    return light_data


def simulate_scene(scene, num_frames, start_x, end_x, max_dimension, height_factor, t, imported_obj):
    """
    Simula la trayectoria del objeto en función de la escena seleccionada.
    :param scene: Escena seleccionada (1: parabólica, 2: caída libre, 3: hacia la cámara)
    :param t: Tiempo normalizado entre 0 y 1 para cada frame
    :param imported_obj: Objeto importado en la escena
    """
    if scene == 1:
        # Movimiento parabólico
        x = start_x + t * (end_x - start_x)
        y = -4 * max_dimension * height_factor * (t - 0.5) ** 2 + max_dimension * height_factor
        imported_obj.location.x = x
        imported_obj.location.z = y

    '''
    elif scene == 2:
        # Caída libre
        x = 0  # Sin movimiento horizontal
        y = max_dimension * height_factor - 9.8 * (t * num_frames / 30)**2  # Caída acelerada
        imported_obj.location.x = x
        imported_obj.location.z = max(0, y)  # Evitar posiciones negativas en Z

    elif scene == 3:
        # Lanzamiento hacia la cámara
        x = 0  # Movimiento en línea recta hacia la cámara
        y = max_dimension * height_factor * (1 - t)  # Z decrece hacia la cámara
        imported_obj.location.y = -t * 5  # Movimiento hacia la cámara
        imported_obj.location.z = y
    '''

    imported_obj.rotation_euler[2] += t * 2 * math.pi / 540  # Rotación continua


def process_object(object_class, base_path, output_folder, orientations_degrees, num_frames, start_x, end_x, height_factor, sensor_width, sensor_height, lens_mm, scene, luminosities, frame_range):
    """Procesa un objeto y genera imágenes para diferentes orientaciones y niveles de luminosidad."""
    obj_path = os.path.join(base_path, f"{object_class}/{object_class}.obj")
    object_folder_path = os.path.join(output_folder, object_class)
    os.makedirs(object_folder_path, exist_ok=True)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.obj(filepath=obj_path)

    imported_obj = bpy.context.selected_objects[0]
    max_dimension = max(imported_obj.dimensions)

    camera_object = setup_camera(sensor_width, sensor_height, lens_mm, max_dimension, height_factor)
    light_data = create_light()

    for luminance in luminosities:
        light_data.energy = luminance
        lum_folder_path = os.path.join(object_folder_path, f'lum{luminance}')
        os.makedirs(lum_folder_path, exist_ok=True)

        for orientation_deg in orientations_degrees:
            orientation_rad = [math.radians(angle) for angle in orientation_deg]
            imported_obj.rotation_euler = orientation_rad

            orientation_str = f"{int(orientation_deg[0])}_{int(orientation_deg[1])}_{int(orientation_deg[2])}"
            orientation_folder_path = os.path.join(lum_folder_path, f'orientation_{orientation_str}')
            os.makedirs(orientation_folder_path, exist_ok=True)

            pose_file_path = os.path.join(orientation_folder_path, f"{object_class}_pose_orientation_{orientation_str}.txt")
            with open(pose_file_path, "w") as pose_file:
                pose_file.write("frame,x,y,z,qx,qy,qz,qw\n")

                for i in range(num_frames):
                    t = i / (num_frames - 1)  # Tiempo normalizado entre 0 y 1
                    simulate_scene(scene, num_frames, start_x, end_x, max_dimension, height_factor, t, imported_obj)

                    bpy.context.view_layer.update()

                    # Usar el rango de frames dependiendo del sensor
                    if frame_range[0] <= i <= frame_range[1]:
                        relative_position = camera_object.matrix_world.inverted_safe() @ imported_obj.matrix_world.translation
                        relative_rotation = camera_object.matrix_world.inverted_safe().to_quaternion() @ imported_obj.matrix_world.to_quaternion()
                        z_distance = abs(relative_position.z)

                        if z_distance > 0:
                            x_pixel = int((relative_position.x / z_distance) * (lens_mm / sensor_width) * bpy.context.scene.render.resolution_x + bpy.context.scene.render.resolution_x / 2)
                            y_pixel = int(-(relative_position.y / z_distance) * (lens_mm / sensor_height) * bpy.context.scene.render.resolution_y + bpy.context.scene.render.resolution_y / 2)

                            pose_file.write(f"{i},{x_pixel},{y_pixel},{z_distance},{relative_rotation.x},{relative_rotation.y},{relative_rotation.z},{relative_rotation.w}\n")

                            output_image_path = os.path.join(orientation_folder_path, f'image_{i:04d}.jpg')
                            bpy.context.scene.render.filepath = output_image_path
                            bpy.ops.render.render(write_still=True)


def main():
    args = parse_arguments()

    sensor_name = args.sensor
    scene = args.scene
    luminosities = args.luminosities
    output_base_folder = os.path.join(args.output_dir, args.sensor, f"{datetime.now().strftime('%Y%m%d_%H%M')}/scene_{scene}/")

    if os.path.exists(output_base_folder):
        shutil.rmtree(output_base_folder)
    os.makedirs(output_base_folder)

    # Cargar configuración de la cámara
    camera_config = load_camera_config(sensor_name)
    h_fov = camera_config['camera']['H-FOV']
    v_fov = camera_config['camera']['V-FOV']
    lens_mm = 50
    sensor_width, sensor_height = calculate_sensor_size(h_fov, v_fov, lens_mm)

    # Configurar ajustes de renderizado
    configure_render_settings()

    orientations_file = "../objects/orientations_24.txt"
    orientations_degrees = np.loadtxt(orientations_file, skiprows=1)

    object_classes = [
        'almohada', 'arbol', 'avion', 'boomerang', 'caja_amarilla', 'caja_azul',
        'carro_rojo', 'clorox', 'dino', 'disco', 'jarron', 'lysoform', 'mobil',
        'paleta', 'pelota', 'sombrero', 'tarro', 'tazon', 'toalla_roja', 'zapatilla'
    ]

    num_frames = 1500
    start_x = -1
    end_x = 1
    height_factor = 1.5

    # Definir los rangos de frames para cada sensor
    sensor_frame_ranges = {
        "evk4": (500, 1000),
        "davis346": (0, 1500),
        #"rog": (200, 1200),  # ejemplo, ajusta según lo que necesites
        #"zed2": (100, 1200)  # ejemplo, ajusta según lo que necesites
    }

    # Obtener el rango de frames según el sensor seleccionado
    frame_range = sensor_frame_ranges.get(sensor_name, (500, 1000))  # Valor por defecto: (500, 1000)

    for object_class in object_classes:
        process_object(object_class, "../objects/all/", output_base_folder, orientations_degrees, num_frames, start_x, end_x, height_factor, sensor_width, sensor_height, lens_mm, scene, luminosities, frame_range)



if __name__ == "__main__":
    main()
