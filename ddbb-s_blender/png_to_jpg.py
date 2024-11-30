import os
import cv2
import argparse

def convert_png_to_jpg(directory):
    """
    Convierte todos los archivos .png a .jpg en el directorio especificado y sus subdirectorios.
    
    Args:
        directory (str): Ruta al directorio principal.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                # Ruta completa del archivo original
                png_path = os.path.join(root, file)
                # Ruta completa del archivo convertido
                jpg_path = os.path.join(root, file[:-4] + '.jpg')
                
                try:
                    # Leer la imagen .png
                    image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
                    if image is None:
                        print(f"Error al leer la imagen: {png_path}")
                        continue
                    
                    # Guardar la imagen como .jpg
                    cv2.imwrite(jpg_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    # Eliminar el archivo .png original si lo deseas
                    os.remove(png_path)
                    
                    #print(f"Convertido: {png_path} -> {jpg_path}")
                except Exception as e:
                    print(f"Error al convertir {png_path}: {e}")

if __name__ == "__main__":
    # Configuración de argparse para recibir el directorio como argumento opcional
    parser = argparse.ArgumentParser(description="Convierte archivos PNG a JPG recursivamente en un directorio.")
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Ruta al directorio que contiene las imágenes PNG."
    )
    
    args = parser.parse_args()
    directory_path = args.directory
    
    # Validar si la ruta existe
    if not os.path.isdir(directory_path):
        print(f"El directorio especificado no existe: {directory_path}")
    else:
        convert_png_to_jpg(directory_path)
