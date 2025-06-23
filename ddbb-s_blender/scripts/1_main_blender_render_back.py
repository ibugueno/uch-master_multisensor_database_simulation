import argparse
import bpy
bpy.ops.preferences.addon_enable(module="io_import_images_as_planes")
import math
import os
import shutil
import yaml
import numpy as np
from datetime import datetime
from mathutils import Vector
import numpy as np
from PIL import Image

def parse_arguments():
    """Define y analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Renderizado y generación de poses 6D.")

    parser.add_argument("--sensor", type=str, choices=["evk4", "davis346", "asus", "zed2"],
                        required=True, help="Selecciona el sensor a usar (evk4, davis346, asus, zed2).")

    parser.add_argument(
        "--scene",
        type=str,
        choices=["0", "1", "2", "3", "all"],
        required=True,
        help="Selecciona la escena: 0 (giro), 1 (parábola), 2 (caída), 3 (hacia cámara), all (todas)"
    )

    parser.add_argument("--luminosities", type=int, nargs='+', choices=[1000], default=[1000],
                        help="Especifica los niveles de luminosidad a simular (por defecto: 1000).")

    parser.add_argument("--background", action="store_true",
                        help="Si se activa, agrega una imagen de fondo detrás del objeto.")

    parser.add_argument("--save_labels", action="store_true", help="Guardar etiquetas de posición normalizada y absoluta")

    parser.add_argument("--output_dir", type=str, default="../data/",
                        help="Directorio base para guardar los datos generados.")

    return parser.parse_args()


def load_camera_config(sensor_name):
    """Carga todos los parámetros necesarios desde YAML."""
    yaml_file = f"../sensors/config_{sensor_name}.yaml"
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    camera = config['camera']
    h_fov = camera['H-FOV']
    v_fov = camera['V-FOV']
    resolution_width = camera['resolution']['width']
    resolution_height = camera['resolution']['height']
    sensor_width_mm = camera['sensor_width_mm']
    sensor_height_mm = camera['sensor_height_mm']
    focal_length_mm_x = camera['focal_length_mm_x']
    focal_length_mm_y = camera['focal_length_mm_y']
    focal_length_px = camera['focal_length_px']  # en x
    focal_length_py = camera['focal_length_py']  # en y

    # Inferir tamaño de píxel promedio (mm por pixel)
    pixel_size_mm_x = sensor_width_mm / resolution_width
    pixel_size_mm_y = sensor_height_mm / resolution_height

    return {
        'h_fov': h_fov,
        'v_fov': v_fov,
        'resolution_width': resolution_width,
        'resolution_height': resolution_height,
        'sensor_width_mm': sensor_width_mm,
        'sensor_height_mm': sensor_height_mm,
        'focal_length_mm_x': focal_length_mm_x,
        'focal_length_mm_y': focal_length_mm_y,
        'focal_length_px': focal_length_px,
        'focal_length_py': focal_length_py,
        'pixel_size_mm_x': pixel_size_mm_x,
        'pixel_size_mm_y': pixel_size_mm_y
    }


def assign_white_material(obj):
    mat = bpy.data.materials.new(name="WhiteMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
        bsdf.inputs["Specular"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 1.0
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def calculate_focal_length(sensor_width_mm, sensor_height_mm, h_fov, v_fov):
    """Calcula la longitud focal a partir del FOV y las dimensiones del sensor."""
    focal_length_h = sensor_width_mm / (2 * math.tan(math.radians(h_fov) / 2))
    focal_length_v = sensor_height_mm / (2 * math.tan(math.radians(v_fov) / 2))
    return focal_length_h, focal_length_v


def render_depth_mask_frame(imported_obj, output_path, binary=False):
    scene = bpy.context.scene
    original_engine = scene.render.engine

    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.color_type = 'OBJECT'

    if not scene.world:
        scene.world = bpy.data.worlds.new("MaskWorld")
    scene.world.use_nodes = False
    scene.world.color = (0.0, 0.0, 0.0)

    for obj in scene.objects:
        if obj.type == 'LIGHT':
            obj.hide_render = True
        elif obj.name != imported_obj.name:
            obj.hide_render = True

    imported_obj.color = (1.0, 1.0, 1.0, 1.0)

    scene.render.filepath = output_path
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.quality = 95

    bpy.ops.render.render(write_still=True)

    for obj in scene.objects:
        obj.hide_render = False

    scene.render.engine = original_engine

    if not binary:
        return

    img = Image.open(output_path).convert("L")
    img_np = np.array(img)
    mask = (img_np > 10).astype(np.uint8) * 255
    bin_img = Image.fromarray(mask)
    bin_img.save(output_path)

    y_indices, x_indices = np.where(mask > 0)

    if len(x_indices) > 0 and len(y_indices) > 0:
        xmin, xmax = x_indices.min(), x_indices.max()
        ymin, ymax = y_indices.min(), y_indices.max()
        w, h = img.size

        bbox_norm = (
            round(xmin / w, 6), round(ymin / h, 6),
            round(xmax / w, 6), round(ymax / h, 6)
        )
        bbox_abs = (xmin, ymin, xmax, ymax)

        # Rutas limpias: det-bbox-norm / det-bbox-abs
        orientation_folder = os.path.basename(os.path.dirname(output_path))           # orientation_...
        lum_folder = os.path.basename(os.path.dirname(os.path.dirname(output_path)))  # lum1000
        scene_folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(output_path))))  # scene_X
        sensor_output_dir = os.path.abspath(os.path.join(os.path.dirname(output_path), "..", "..", "..", ".."))

        norm_path = os.path.join(sensor_output_dir, "det-bbox-norm", scene_folder, lum_folder, orientation_folder)
        abs_path = os.path.join(sensor_output_dir, "det-bbox-abs", scene_folder, lum_folder, orientation_folder)

        os.makedirs(norm_path, exist_ok=True)
        os.makedirs(abs_path, exist_ok=True)

        file_name = os.path.basename(output_path).replace(".jpg", ".txt")
        norm_txt = os.path.join(norm_path, file_name)
        abs_txt = os.path.join(abs_path, file_name)

        with open(norm_txt, "w") as f:
            f.write("xmin_norm,ymin_norm,xmax_norm,ymax_norm\n")
            f.write(",".join(map(str, bbox_norm)) + "\n")

        with open(abs_txt, "w") as f:
            f.write("xmin,ymin,xmax,ymax\n")
            f.write(",".join(map(str, bbox_abs)) + "\n")




def configure_render_settings():
    """Configura las preferencias de renderizado para GPU."""
    #bpy.context.scene.render.engine = 'CYCLES'
    #bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    #print("bpy.context.scene.render.engine: ", bpy.context.scene.render.engine)

    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'METAL'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        device.use = True


def setup_camera(sensor_width_mm, sensor_height_mm, h_fov, v_fov, max_dimension, height_factor):
    """Crea y configura una cámara."""
    camera_data = bpy.data.cameras.new(name='Camera')
    camera_object = bpy.data.objects.new('Camera', camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    camera_object.location = (0, -0.9, max_dimension * height_factor)
    camera_object.rotation_euler = (math.pi / 2, 0, 0)
    bpy.context.scene.camera = camera_object

    focal_length_h, focal_length_v = calculate_focal_length(sensor_width_mm, sensor_height_mm, h_fov, v_fov)
    camera_data.lens = focal_length_h
    camera_data.sensor_width = sensor_width_mm
    camera_data.sensor_height = sensor_height_mm  
    return camera_object


def create_light(camera_object):
    """Crea una luz AREA suave colocada detrás de la cámara, en diagonal desde la izquierda."""
    # Crear luz tipo AREA
    light_data = bpy.data.lights.new(name="AreaLight", type='AREA')
    light_data.shape = 'RECTANGLE'
    light_data.size = 8.0 #6
    light_data.size_y = 8.0 #6

    # Crear objeto de luz
    light_object = bpy.data.objects.new(name="AreaLight", object_data=light_data)
    bpy.context.collection.objects.link(light_object)

    # Offset para colocar la luz detrás y a la izquierda de la cámara
    back_offset = Vector((0, -2, 0))  # -X (izquierda), -Y (detrás)
    light_object.location = camera_object.location + back_offset

    # Dirección hacia la cámara (o hacia el centro de la escena si prefieres)
    direction = (camera_object.location - light_object.location).normalized()
    rotation_quat = direction.to_track_quat('-Z', 'Y')
    light_object.rotation_euler = rotation_quat.to_euler()

    return light_data




def simulate_scene(scene, num_frames, max_dimension, height_factor, t, imported_obj):
    """Simula la trayectoria del objeto según la escena seleccionada."""

    # Define una rotación uniforme por frame (2 vueltas en total)
    total_rotations = 3  # vueltas completas
    rotation_per_frame = total_rotations * 2 * math.pi / num_frames

    if scene == 0: # Mantener el objeto centrado y girando sobre su propio eje
        
        imported_obj.location.x = 0  # Centrado en X
        imported_obj.location.z = max_dimension * height_factor  # Centrado verticalmente
        imported_obj.location.y = 0

        imported_obj.rotation_euler[2] += rotation_per_frame

    elif scene == 1: # Lanzamiento perpendicular a la camara
        
        start_x, end_x = -1, 1
        x = start_x + t * (end_x - start_x)
        y = -4 * max_dimension * height_factor * (t - 0.5) ** 2 + max_dimension * height_factor
        imported_obj.location.x, imported_obj.location.y, imported_obj.location.z = x, 0, y
    
        imported_obj.rotation_euler[2] += t * 2 * math.pi / 540

    elif scene == 2: # Caida libre
        
        y = max_dimension * height_factor * 2 - 9.8 * (t * num_frames / 1000)**2  # Caída acelerada
        imported_obj.location.x, imported_obj.location.y, imported_obj.location.z = 0, 0, y

        imported_obj.rotation_euler[2] += rotation_per_frame

    elif scene == 3: #Lanzamiento a la camara

        start_x, end_x = 2.6, -0.4
        x = start_x + t * (end_x - start_x)
        y = -3 * max_dimension * height_factor * (t - 0.5) ** 2 + max_dimension * (1.5*height_factor)
        imported_obj.location.x, imported_obj.location.y, imported_obj.location.z = 0, x, y
    
        imported_obj.rotation_euler[2] += t * 2 * math.pi / 540


def setup_object_index_compositor(output_dir):
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new('CompositorNodeRLayers')
    id_mask = tree.nodes.new('CompositorNodeIDMask')
    id_mask.index = 1
    file_output = tree.nodes.new('CompositorNodeOutputFile')
    file_output.base_path = output_dir
    file_output.file_slots[0].path = "mask_####"

    tree.links.new(rl.outputs['IndexOB'], id_mask.inputs['ID value'])
    tree.links.new(id_mask.outputs['Alpha'], file_output.inputs[0])

    return file_output  



def process_object(object_class, base_path, sensor_output_dir, orientations_degrees, num_frames, height_factor,
                   resolution_width, resolution_height, h_fov, v_fov, lens_mm, sensor_width_mm,
                   sensor_height_mm, lens_px, scene, luminosities, frame_range, use_background, save_labels):

    obj_path = os.path.join(base_path, f"{object_class}/{object_class}.obj")

    bpy.context.view_layer.use_pass_object_index = True
    file_output_node = setup_object_index_compositor(sensor_output_dir)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.obj(filepath=obj_path)

    imported_obj = bpy.context.selected_objects[0]
    imported_obj.pass_index = 1

    max_dimension = max(imported_obj.dimensions)
    camera_object = setup_camera(sensor_width_mm, sensor_height_mm, h_fov, v_fov, max_dimension, height_factor)

    if use_background:
        background_path = "../background/amtc_2.jpg"
        add_background_image(background_path, camera_object)

    light_data = create_light(camera_object)
    bpy.context.scene.render.resolution_x = resolution_width
    bpy.context.scene.render.resolution_y = resolution_height

    scene_str = f"scene_{scene}"

    for luminance in luminosities:
        light_data.energy = luminance
        lum_str = f"lum{luminance}"

        for orientation_deg in orientations_degrees:
            orientation_str = f"orientation_{int(orientation_deg[0])}_{int(orientation_deg[1])}_{int(orientation_deg[2])}"
            orientation_rad = [math.radians(angle) for angle in orientation_deg]
            imported_obj.rotation_euler = orientation_rad

            base_subpath = os.path.join(scene_str, lum_str, orientation_str)

            image_output_path = os.path.join(sensor_output_dir, "images", base_subpath)
            depth_mask_output_path = os.path.join(sensor_output_dir, "masks-depth", base_subpath)
            seg_mask_output_path = os.path.join(sensor_output_dir, "masks-seg", base_subpath)
            pose6d_abs_output_path = os.path.join(sensor_output_dir, "pose6d-abs", base_subpath)
            pose6d_norm_output_path = os.path.join(sensor_output_dir, "pose6d-norm", base_subpath)

            os.makedirs(image_output_path, exist_ok=True)

            if save_labels:
                os.makedirs(depth_mask_output_path, exist_ok=True)
                os.makedirs(seg_mask_output_path, exist_ok=True)
                os.makedirs(pose6d_abs_output_path, exist_ok=True)
                os.makedirs(pose6d_norm_output_path, exist_ok=True)

            file_output_node.base_path = depth_mask_output_path

            for i in range(num_frames):
                t = i / (num_frames - 1)
                simulate_scene(scene, num_frames, max_dimension, height_factor, t, imported_obj)
                bpy.context.view_layer.update()

                if frame_range[0] <= i <= frame_range[1]:
                    rel_pos = camera_object.matrix_world.inverted_safe() @ imported_obj.matrix_world.translation
                    rel_rot = camera_object.matrix_world.inverted_safe().to_quaternion() @ imported_obj.matrix_world.to_quaternion()
                    z_m = abs(rel_pos.z)

                    if z_m > 0:
                        x_px = int((rel_pos.x / z_m) * lens_px + resolution_width / 2)
                        y_px = int(-(rel_pos.y / z_m) * lens_px + resolution_height / 2)

                        if 0 <= x_px < resolution_width and 0 <= y_px < resolution_height:
                            x_norm = round(x_px / resolution_width, 6)
                            y_norm = round(y_px / resolution_height, 6)
                            z_cm = round(z_m * 100, 6)

                            qw, qx, qy, qz = (
                                round(rel_rot.w, 6),
                                round(rel_rot.x, 6),
                                round(rel_rot.y, 6),
                                round(rel_rot.z, 6),
                            )

                            if save_labels:
                                abs_file = os.path.join(pose6d_abs_output_path, f"image_{i:04d}.txt")
                                norm_file = os.path.join(pose6d_norm_output_path, f"image_{i:04d}.txt")

                                with open(abs_file, "w") as pf:
                                    pf.write("x_px,y_px,z_cm,qw,qx,qy,qz\n")
                                    pf.write(f"{x_px},{y_px},{z_cm},{qw},{qx},{qy},{qz}\n")

                                with open(norm_file, "w") as pf:
                                    pf.write("x_norm,y_norm,z_cm,qw,qx,qy,qz\n")
                                    pf.write(f"{x_norm},{y_norm},{z_cm},{qw},{qx},{qy},{qz}\n")

                                output_depth = os.path.join(depth_mask_output_path, f"image_{i:04d}.jpg")
                                output_seg = os.path.join(seg_mask_output_path, f"image_{i:04d}.jpg")

                                render_depth_mask_frame(imported_obj, output_depth, binary=False)
                                render_depth_mask_frame(imported_obj, output_seg, binary=True)

                            output_img = os.path.join(image_output_path, f'image_{i:04d}.jpg')
                            bpy.context.scene.render.filepath = output_img
                            bpy.context.scene.render.image_settings.file_format = 'JPEG'
                            bpy.context.scene.render.image_settings.quality = 95
                            bpy.context.scene.render.image_settings.color_mode = 'RGB'
                            bpy.ops.render.render(write_still=True)


            #break
        #break


def make_plane_shadeless(plane, image_path):
    """Convierte un plano en fondo que no recibe luz ni sombra, con imagen como textura."""

    mat = bpy.data.materials.new(name="BackgroundShadeless")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Limpiar nodos existentes
    for node in nodes:
        nodes.remove(node)

    # Nodos necesarios
    tex_image = nodes.new(type='ShaderNodeTexImage')
    emission = nodes.new(type='ShaderNodeEmission')
    output = nodes.new(type='ShaderNodeOutputMaterial')

    # Cargar imagen del fondo
    img = bpy.data.images.load(image_path)
    tex_image.image = img

    # Conectar imagen → emisión → salida
    links.new(tex_image.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])

    # Asignar material al plano
    plane.data.materials.clear()
    plane.data.materials.append(mat)

    # Opcional: asegurar que no reciba sombras
    plane.cycles.is_shadow_catcher = False
    plane.cycles.use_shadow_cast = False


def add_background_image(image_path, camera_object):
    """Agrega una imagen como plano detrás del objeto, centrada con la cámara."""

    # Importar imagen como plano
    bpy.ops.import_image.to_plane(files=[{"name": os.path.basename(image_path)}],
                                  directory=os.path.dirname(image_path),
                                  relative=False)

    bg_plane = bpy.context.selected_objects[0]

    # Ubicar plano detrás del objeto
    #bg_plane.location = (camera_object.location.x, 0.9, camera_object.location.z)
    bg_plane.location = (camera_object.location.x, 3, camera_object.location.z)
    bg_plane.rotation_euler = (math.pi / 2, 0, 0)

    bg_plane.scale = (3.5, 3.5, 1)  # Ajusta según tus necesidades

    # Aplicar material shadeless
    make_plane_shadeless(bg_plane, image_path)


def main():
    args = parse_arguments()

    sensor_name = args.sensor
    luminosities = args.luminosities

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    ddbb_root = os.path.join(args.output_dir, f"ddbb-s-{timestamp}")
    sensor_output_dir = os.path.join(ddbb_root, args.sensor)

    if os.path.exists(sensor_output_dir):
        shutil.rmtree(sensor_output_dir)
    os.makedirs(sensor_output_dir, exist_ok=True)

    cam_config = load_camera_config(sensor_name)

    h_fov = cam_config['h_fov']
    v_fov = cam_config['v_fov']
    resolution_width = cam_config['resolution_width']
    resolution_height = cam_config['resolution_height']
    sensor_width_mm = cam_config['sensor_width_mm']
    sensor_height_mm = cam_config['sensor_height_mm']
    focal_length_mm_x = cam_config['focal_length_mm_x']
    focal_length_mm_y = cam_config['focal_length_mm_y']
    focal_length_px = cam_config['focal_length_px']
    focal_length_py = cam_config['focal_length_py']
    pixel_size_mm_x = cam_config['pixel_size_mm_x']
    pixel_size_mm_y = cam_config['pixel_size_mm_y']

    configure_render_settings()

    orientations_file = "../objects/orientations_24.txt"
    orientations_degrees = np.loadtxt(orientations_file, skiprows=1)

    '''
    object_classes = [
        'almohada', 'arbol', 'avion', 'boomerang', 'caja_amarilla', 'caja_azul',
        'carro_rojo', 'clorox', 'dino', 'disco', 'jarron', 'lysoform', 'mobil',
        'paleta', 'pelota', 'sombrero', 'tarro', 'tazon', 'toalla_roja', 'zapatilla'
    ]
    '''
    
    object_classes = [
        'almohada', 'arbol', 'avion', 'boomerang', 'caja_amarilla', 'caja_azul',
        'carro_rojo', 'clorox', 'dino', 'jarron', 'lysoform', 'mobil',
        'paleta', 'pelota', 'tarro', 'sombrero', 'zapatilla'
    ]

    #object_classes = ['jarron']


    num_frames = 1500
    height_factor = 1.5

    #'''
    sensor_frame_ranges = {
        "evk4":     {0: (0, 1500), 1: (0, 1500), 2: (0, 1500), 3: (900, 1500)},
        "davis346": {0: (0, 1500), 1: (0, 1500), 2: (0, 1500), 3: (900, 1500)},
        "asus":     {0: (0, 1500), 1: (0, 1500), 2: (0, 1500), 3: (900, 1500)},
        "zed2":     {0: (0, 1500), 1: (0, 1500), 2: (0, 1500), 3: (900, 1500)}
    }
    
    '''

    sensor_frame_ranges = {
        "evk4":     {0: (0,0), 1: (0,0), 2: (0,0), 3: (900,900)},
        "davis346": {0: (0,0), 1: (0,0), 2: (0,0), 3: (0,0)},
        "asus":     {0: (0,0), 1: (0,0), 2: (0,0), 3: (0,0)}, 
    }
    '''

    # Escenas a ejecutar
    if args.scene == "all":
        scenes_to_run = [0, 1, 2, 3]
    else:
        scenes_to_run = [int(args.scene)]

    for scene in scenes_to_run:
        frame_range = sensor_frame_ranges.get(sensor_name, {}).get(scene, (500, 1000))
        print(f"Procesando escena {scene} con rango {frame_range}")

        for object_class in object_classes:
            process_object(
                object_class, "../objects/all/", sensor_output_dir, orientations_degrees,
                num_frames, height_factor, resolution_width, resolution_height, h_fov, v_fov,
                focal_length_mm_x, sensor_width_mm, sensor_height_mm, focal_length_px,
                scene, luminosities, frame_range, args.background, args.save_labels
            )






if __name__ == "__main__":
    main()
