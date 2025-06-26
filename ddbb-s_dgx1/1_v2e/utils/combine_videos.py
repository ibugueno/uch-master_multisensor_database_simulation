import cv2

def combinar_videos_vertical(video1_path, video2_path, salida_path):
    # Abrir ambos videos
    video1 = cv2.VideoCapture(video1_path)
    video2 = cv2.VideoCapture(video2_path)

    # Obtener las propiedades del video (ancho, alto, FPS)
    ancho = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video1.get(cv2.CAP_PROP_FPS))

    # Crear el objeto VideoWriter para guardar el video combinado
    salida = cv2.VideoWriter(
        salida_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps*4,
        (ancho, alto * 2)  # El nuevo alto es la suma de ambos videos
    )

    while True:
        # Leer un frame de cada video
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        # Si alguno de los videos termina, detener el bucle
        if not ret1 or not ret2:
            break

        # Concatenar los frames verticalmente
        frame_combinado = cv2.vconcat([frame1, frame2])

        # Mostrar el frame combinado (opcional)
        cv2.imshow('Video Combinado', frame_combinado)

        # Escribir el frame combinado en el archivo de salida
        salida.write(frame_combinado)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los objetos de video y cerrar ventanas
    video1.release()
    video2.release()
    salida.release()
    cv2.destroyAllWindows()

# Rutas de los videos y archivo de salida
video1_path = "../output/almohada_back_davis346/rgb_video.mp4"  # Cambia esta ruta
video2_path = "../output/almohada_back_davis346/output_timestamp_image_aedat4.mp4"  # Cambia esta ruta
salida_path = "../output/almohada_back_davis346/output_frames_events.mp4"

# Ejecutar la función
combinar_videos_vertical(video1_path, video2_path, salida_path)
