# Monitorización de Trabajo con Reconocimiento Facial

## Descripción

Este proyecto es un sistema de monitorización de trabajo que utiliza reconocimiento facial y detección de objetos para seguir la presencia y actividad de una persona específica.
El sistema puede detectar cuándo estas modo focus, en pausa o utilizando el teléfono móvil, proporcionando métricas en tiempo real sobre su tiempo de enfoque, 
pausa y uso del teléfono.



## Características

- **Reconocimiento Facial Personalizado:** Calibración inicial para reconocer automáticamente a Pedro ( si te trabajas de lado, mueve la cabeza en la calibración)
- **Detección de Objetos:** Utiliza el modelo YOLO para identificar objetos específicos como botellas y vasos
- **Gestión de Estados:**
  - **Focus Time:** Tiempo total que Pedro está enfocado en su trabajo.
  - **Pause Time:** Tiempo total que Pedro está en pausa (no enfocado o utilizando el teléfono).
  - **Phone Time:** Tiempo total que Pedro utiliza el teléfono móvil.
  - **Inicio de Focus Time:** Hora exacta en la que Pedro comenzó a estar enfocado.
- **Interfaz Visual:** Muestra en tiempo real los contadores de Focus Time, Pause Time, Phone Time y la hora de inicio del Focus Time directamente en la ventana de video.
- **Seguimiento Corporal:** Integración con MediaPipe Pose para mantener la detección de Pedro incluso cuando gira su rostro.

## Requisitos

- **Sistema Operativo:** Windows, macOS o Linux.
- **Lenguaje de Programación:** Python 3.10 
- **Dependencias:**
  - `opencv-python`
  - `numpy`
  - `mediapipe`
  - `face_recognition`
  - `dlib` (requerido por `face_recognition`)

- 
## Instalación

1. **Clonar el Repositorio:**

   ```bash
   git clone https://github.com/tu-usuario/monitorizacion-trabajo.git
   cd monitorizacion-trabajo
2. **Modelo YOLO:** 
- Crear carpeta yolo 
- Descargar los archivos `yolov3.weights`, `yolov3.cfg` y `coco.names` desde https://pjreddie.com/darknet/yolo/