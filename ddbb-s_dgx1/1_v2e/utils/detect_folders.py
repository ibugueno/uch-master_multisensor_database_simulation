import os

# Ruta raíz
root_dir = 'input/frames_for_generate_val_with_back_without_blur/3_ddbs-s_with_back_without_blur'

# Lista para almacenar todas las rutas completas de los directorios de orientación
orientation_dirs = []

# Archivo de salida
output_file = 'data/files_frames_for_generate_val_with_back_without_blur.txt'

# Recorrer recursivamente la estructura de directorios
for subdir1 in ['asus','zed2','davis346', 'evk4']:  # Nivel 1: 'davis346', 'evk4'
    subdir1_path = os.path.join(root_dir, subdir1)
    if os.path.isdir(subdir1_path):
        for subdir2 in os.listdir(subdir1_path):  # Nivel 2: 'evk4_scn2_lum9'
            subdir2_path = os.path.join(subdir1_path, subdir2)
            if os.path.isdir(subdir2_path):
                for scene_dir in os.listdir(subdir2_path):  # Nivel 3: 'scene_2'
                    scene_dir_path = os.path.join(subdir2_path, scene_dir)
                    if os.path.isdir(scene_dir_path):
                        for obj_dir in os.listdir(scene_dir_path):  # Nivel 4: 'avion', 'tarro'
                            obj_dir_path = os.path.join(scene_dir_path, obj_dir)
                            if os.path.isdir(obj_dir_path):
                                for subfolder in os.listdir(obj_dir_path):  # Nivel 5: 'lum9'
                                    subfolder_path = os.path.join(obj_dir_path, subfolder)
                                    if os.path.isdir(subfolder_path):
                                        for orientation_dir in os.listdir(subfolder_path):  # Nivel 6: 'orientation_x_y_z'
                                            orientation_dir_path = os.path.join(subfolder_path, orientation_dir)
                                            if os.path.isdir(orientation_dir_path):  # Verificar que existe
                                                # Remover el prefijo 'input/' de la ruta
                                                relative_path = os.path.relpath(orientation_dir_path, root_dir)
                                                orientation_dirs.append(relative_path)

# Guardar las rutas en un archivo de texto
with open(output_file, 'w') as f:
    f.write('\n'.join(orientation_dirs))

# Imprimir las rutas generadas
if orientation_dirs:
    print(f"Se encontraron {len(orientation_dirs)} directorios de orientación. Guardados en {output_file}")
    #for path in orientation_dirs:
    #    print(path)
else:
    print("No se encontraron directorios de orientación.")
