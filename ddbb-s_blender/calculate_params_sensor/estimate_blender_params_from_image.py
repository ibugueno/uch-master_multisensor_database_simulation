import math
import yaml
import argparse
import os

def main():
    # === 1. Parsear argumento del sensor ===
    parser = argparse.ArgumentParser(description="Cálculo de FOV y parámetros compatibles con Blender.")
    parser.add_argument("--sensor", type=str, required=True, help="Nombre del sensor (ej. 'asus')")
    args = parser.parse_args()
    sensor_name = args.sensor

    # === 2. Cargar YAML de parámetros reales ===
    input_yaml_path = f"yaml/sensor_params_{sensor_name}.yaml"
    if not os.path.exists(input_yaml_path):
        raise FileNotFoundError(f"No se encontró el archivo: {input_yaml_path}")

    with open(input_yaml_path, "r") as file:
        data = yaml.safe_load(file)

    # === 3. Extraer resolución e intrínsecos ===
    width_px = data["image_width"]
    height_px = data["image_height"]
    fx = data["camera_matrix"]["data"][0]
    fy = data["camera_matrix"]["data"][4]

    # === 4. Calcular FOVs iniciales (a partir de parámetros de calibración) ===
    fov_h = 2 * math.degrees(math.atan(width_px / (2 * fx)))
    fov_v = 2 * math.degrees(math.atan(height_px / (2 * fy)))

    # === 5. Calcular tamaño del sensor estimado en mm (asumiendo tamaño cuadrado de píxel) ===
    if sensor_name == "asus":
        pixel_size_mm = 0.003  # ASUS ROG Eye S (~3 µm)
    elif sensor_name == "davis346":
        pixel_size_mm = 0.0185  # DAVIS346: 18.5 µm
    elif sensor_name == "evk4":
        pixel_size_mm = 0.00486  # EVK4 (IMX636): 4.86 µm
    elif sensor_name == "zed2":
        pixel_size_mm = 0.0022  # ZED2: 2.2 µm
    else:
        pixel_size_mm = 1e-3  # Placeholder

    # Tamaño estimado del sensor (mm)
    sensor_width_mm = width_px * pixel_size_mm
    sensor_height_mm = height_px * pixel_size_mm

    # Focal length físico estimado en mm
    focal_length_mm_x = fx * pixel_size_mm
    focal_length_mm_y = fy * pixel_size_mm

    # === Corrección SOLO para EVK4 (objeto se ve más grande de lo esperado) ===
    if sensor_name == "evk4":
        # Estimación: el objeto aparece ~1.12x más ancho y ~1.18x más alto de lo debido
        factor_width = 305 / 260
        factor_height = 629 / 532
        correction_factor = 1 / ((factor_width + factor_height) / 2)  # ≈ 0.87

        focal_length_mm_x *= correction_factor
        focal_length_mm_y *= correction_factor

        # Recalcular FOV con focal ajustado
        fov_h = 2 * math.degrees(math.atan(sensor_width_mm / (2 * focal_length_mm_x)))
        fov_v = 2 * math.degrees(math.atan(sensor_height_mm / (2 * focal_length_mm_y)))

    # Reafirmar focal en pixeles
    focal_length_px = fx
    focal_length_py = fy


    # === 6. Mostrar resultados ===
    print("\n===== PARÁMETROS COMPATIBLES CON BLENDER =====")
    print(f"Resolución: {width_px}x{height_px} px")
    print(f"Focal length (px): fx = {fx:.2f}, fy = {fy:.2f}")
    print(f"Focal length (mm): {focal_length_mm_x:.4f} x {focal_length_mm_y:.4f}")
    print(f"Sensor size (mm): {sensor_width_mm:.2f} x {sensor_height_mm:.2f}")
    print(f"H-FOV: {fov_h:.2f}°, V-FOV: {fov_v:.2f}°")

    # === 7. Guardar YAML de salida ===
    output_yaml_path = os.path.join("../sensors/", f"config_{sensor_name}.yaml")

    config = {
        'camera': {
            'resolution': {
                'width': width_px,
                'height': height_px
            },
            'H-FOV': round(fov_h, 2),
            'V-FOV': round(fov_v, 2),
            'sensor_width_mm': round(sensor_width_mm, 2),
            'sensor_height_mm': round(sensor_height_mm, 2),
            'focal_length_mm_x': round(focal_length_mm_x, 4),
            'focal_length_mm_y': round(focal_length_mm_y, 4),
            'focal_length_px': round(fx, 2),
            'focal_length_py': round(fy, 2)
        }
    }

    with open(output_yaml_path, "w") as file:
        yaml.dump(config, file)

    print(f"\n✅ Archivo YAML guardado como '{output_yaml_path}'")

if __name__ == "__main__":
    main()
