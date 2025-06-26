import cv2
import os

def generate_video_from_images(image_folder, output_video_path, fps=30):
    # Obtener la lista de archivos de imagen, ordenada alfabéticamente
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    
    if not images:
        print("No se encontraron imágenes en la carpeta.")
        return

    # Leer la primera imagen para obtener dimensiones
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Configurar el objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Leer y agregar cada imagen al video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video guardado en: {output_video_path}")

# Carpeta de imágenes
image_folder = "../input/almohada_back_davis346/"  # Ruta a la carpeta con las imágenes
output_video_path = "../output/almohada_back_davis346/rgb_video.mp4"  # Nombre del archivo de salida

# Generar el video
generate_video_from_images(image_folder, output_video_path, fps=30)
