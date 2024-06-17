import bpy

# Configurar Blender para usar Cycles y la GPU (CUDA)
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

# Obtener y activar dispositivos CUDA
bpy.context.preferences.addons['cycles'].preferences.get_devices()
for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    if device.type == 'CUDA':
        device.use = True
        print(f"Usando dispositivo GPU: {device.name}")

# Verificar que la GPU está seleccionada en todas las escenas
for scene in bpy.data.scenes:
    scene.cycles.device = 'GPU'

# Crear una escena de prueba para renderizar
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
bpy.ops.object.camera_add(location=(0, -3, 3), rotation=(1.1, 0, 0))
bpy.context.scene.camera = bpy.context.object

# Configurar la renderización
bpy.context.scene.render.image_settings.file_format = 'JPEG'
bpy.context.scene.render.filepath = '/tmp/test_render.jpg'
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080

# Renderizar la imagen
bpy.ops.render.render(write_still=True)

print("Renderizado completado. Verifica la imagen en /tmp/test_render.jpg")
