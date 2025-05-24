# app.py
import subprocess
import sys
import os
import cv2
import datetime
import threading
from ultralytics import YOLO
import time

from flask import Flask, render_template, Response, request, jsonify, redirect, url_for

# --- Variables Globales y Bloqueos ---
GLOBAL_COUNTS_LOCK = threading.Lock()
GLOBAL_CLASS_COUNTS = {}
CAMERA_ACTIVE = False
SAVE_VIDEO = False
VIDEO_WRITER = None
OUTPUT_VIDEO_PATH = None
RECORDING_STARTED_AT = None
UNKNOWN_CONF_THRESHOLD = 0.50 # Umbral para considerar una detección como 'desconocido'

# --- Rutas de Carpetas ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ¡CORRECCIÓN CRÍTICA DE LA RUTA DEL MODELO!
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'object_detection', 'best.pt')

# Rutas de subida
UPLOAD_FOLDER_BASE = os.path.join(BASE_DIR, 'static', 'uploads')
UPLOAD_IMAGES_FOLDER = os.path.join(UPLOAD_FOLDER_BASE, 'images')
UPLOAD_VIDEOS_FOLDER = os.path.join(UPLOAD_FOLDER_BASE, 'videos')

# ¡CORRECCIÓN CRÍTICA DE LA RUTA DEL PLACEHOLDER!
# Asegúrate de que este archivo exista en static/images/
PLACEHOLDER_IMAGE_PATH = os.path.join(BASE_DIR, 'static', 'images', 'cargando_video.png')
print(f"DEBUG: Ruta de placeholder calculada: {PLACEHOLDER_IMAGE_PATH}")

# --- Inicialización de Carpetas ---
os.makedirs(UPLOAD_IMAGES_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_VIDEOS_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'static', 'images'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'static', 'logos'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'static', 'css'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'static', 'flags'), exist_ok=True)


# --- Servicio de Detección ---
class DetectionService:
    def __init__(self):
        print(f"[INFO] Cargando modelo YOLO desde: {MODEL_PATH}")
        try:
            self.model = YOLO(MODEL_PATH)
            self.class_names = self.model.names
            print(f"[INFO] Modelo cargado. Clases detectadas: {self.class_names}")

            with GLOBAL_COUNTS_LOCK:
                for name_id, name_label in self.class_names.items():
                    GLOBAL_CLASS_COUNTS[name_label] = 0
                GLOBAL_CLASS_COUNTS['desconocido'] = 0 # Inicializar 'desconocido' también
            print(f"[INFO] Contadores globales inicializados con: {GLOBAL_CLASS_COUNTS.keys()}")

        except Exception as e:
            print(f"[ERROR] No se pudo cargar el modelo YOLO desde {MODEL_PATH}: {e}")
            self.model = None
            self.class_names = {}

    def process_frame(self, frame, update_global_counts=True):
        """
        Procesa un frame de video para detección de objetos.
        Args:
            frame (numpy.array): El frame de imagen a procesar.
            update_global_counts (bool): Si es True, actualiza los contadores globales.
        Returns:
            tuple: (drawn_frame, frame_detections)
        """
        if self.model is None:
            print("[ERROR] Modelo YOLO no cargado. No se puede procesar el frame.")
            all_possible_classes = list(self.class_names.values()) + ['desconocido']
            return frame.copy(), {name: 0 for name in all_possible_classes}

        results = self.model(frame, imgsz=640, conf=0.1, verbose=False)[0]
        drawn_frame = frame.copy()
        frame_detections = {name: 0 for name in self.class_names.values()}
        frame_detections['desconocido'] = 0
        
        for result in results.boxes:
            cls_id = int(result.cls[0])
            confidence = float(result.conf[0])
            xyxy = result.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy[:4]

            label = ""
            color = (0, 255, 0) # Verde por defecto para detecciones válidas

            if confidence < UNKNOWN_CONF_THRESHOLD:
                label = f"desconocido {confidence:.2f}"
                color = (0, 0, 255) # Rojo para 'desconocido'
                frame_detections['desconocido'] += 1
            else:
                name = self.class_names.get(cls_id, 'desconocido')
                label = f"{name} {confidence:.2f}"
                color = (0, 255, 0)
                frame_detections[name] += 1
            
            cv2.rectangle(drawn_frame, (x1, y1), (x2, y2), color, 2)
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(drawn_frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if update_global_counts:
            with GLOBAL_COUNTS_LOCK:
                for class_name, count in frame_detections.items():
                    GLOBAL_CLASS_COUNTS[class_name] += count

        return drawn_frame, frame_detections

detection_service = DetectionService()


# --- Servicio de Stream de Cámara ---
class CameraStream:
    def __init__(self):
        self.PLACEHOLDER_BYTES = b''
        self._load_placeholder()
        self.cap = None
        self.is_camera_connected = False

    def _load_placeholder(self):
        """Carga y redimensiona la imagen de placeholder."""
        try:
            placeholder_frame = cv2.imread(PLACEHOLDER_IMAGE_PATH)
            if placeholder_frame is None:
                raise FileNotFoundError(f"Placeholder image not found at {PLACEHOLDER_IMAGE_PATH}")
            
            # Redimensionar el placeholder a un tamaño fijo (ej. 640x480)
            # Esto es importante para que el tamaño del placeholder y del video sea consistente
            placeholder_frame = cv2.resize(placeholder_frame, (640, 480)) 

            _, placeholder_buffer = cv2.imencode('.jpg', placeholder_frame)
            self.PLACEHOLDER_BYTES = placeholder_buffer.tobytes()
            print(f"[INFO] Imagen de placeholder cargada desde: {PLACEHOLDER_IMAGE_PATH}")
        except Exception as e:
            print(f"[ERROR] No se pudo cargar la imagen de placeholder desde {PLACEHOLDER_IMAGE_PATH}: {e}. Asegúrate de que el archivo existe y es una imagen válida.")
            self.PLACEHOLDER_BYTES = b''

    def get_counts(self):
        """Retorna una copia de los conteos globales."""
        with GLOBAL_COUNTS_LOCK:
            return GLOBAL_CLASS_COUNTS.copy()

    def generate_frames(self):
        """Generador de frames para el stream de video."""
        global CAMERA_ACTIVE
        global SAVE_VIDEO, VIDEO_WRITER, OUTPUT_VIDEO_PATH, RECORDING_STARTED_AT

        # Intenta conectar la cámara solo una vez al inicio del stream o si se ha desconectado
        # Esta lógica asegura que no se intenta abrir la cámara en cada ciclo si ya está abierta
        if not self.is_camera_connected:
            self.cap = cv2.VideoCapture(0) # Intenta abrir la cámara con índice 0
            time.sleep(1) # Dale un segundo a la cámara para inicializarse

            if not self.cap.isOpened():
                print("[WARNING] No se pudo abrir la cámara. Sirviendo imagen de placeholder.")
                self.is_camera_connected = False
                CAMERA_ACTIVE = False
                # Aquí no reiniciamos conteos globales, ya que no hay cámara real activa
            else:
                print("[INFO] Cámara conectada y activa.")
                self.is_camera_connected = True
                CAMERA_ACTIVE = True
                
                # Reiniciar conteos globales solo cuando la cámara se conecta (o reconecta) exitosamente
                with GLOBAL_COUNTS_LOCK:
                    for key in GLOBAL_CLASS_COUNTS:
                        GLOBAL_CLASS_COUNTS[key] = 0
                
                # Configuración inicial del grabador de video (solo cuando la cámara está activa)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps == 0: # Fallback si CAP_PROP_FPS retorna 0
                    fps = 20.0 
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.video_dims = (width, height) # Guardar dimensiones para el grabador
                self.video_fps = fps
                self.video_fourcc = fourcc

                print(f"[INFO] Iniciando stream de cámara. Resolución: {width}x{height}, FPS: {fps}")

        # Bucle principal de transmisión de frames
        while True:
            if not self.is_camera_connected:
                # Si la cámara no está conectada, enviamos el placeholder.
                # No procesamos el placeholder con el modelo de detección.
                if self.PLACEHOLDER_BYTES:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + self.PLACEHOLDER_BYTES + b'\r\n')
                time.sleep(1) # Espera antes de volver a intentar o enviar el placeholder

                # Intentar reconectar la cámara periódicamente si no está activa
                # Esto es crucial para la reconexión automática si el usuario inicia DroidCam más tarde
                if not self.cap or not self.cap.isOpened(): # Si cap es None o no está abierto
                    self.cap = cv2.VideoCapture(0) # Intenta abrir la cámara de nuevo
                    time.sleep(0.5) # Pequeña pausa antes de verificar
                    if self.cap.isOpened():
                        print("[INFO] Cámara reconectada exitosamente.")
                        self.is_camera_connected = True
                        CAMERA_ACTIVE = True
                        # Reiniciar conteos al reconectar
                        with GLOBAL_COUNTS_LOCK:
                            for key in GLOBAL_CLASS_COUNTS:
                                GLOBAL_CLASS_COUNTS[key] = 0
                        # Recargar dimensiones y FPS si reconecta (pueden cambiar)
                        self.video_dims = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
                        self.video_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                continue # Continúa al siguiente ciclo (ya sea para enviar placeholder o intentar reconectar)

            # Si la cámara está conectada y activa
            success, frame = self.cap.read()
            if not success:
                print("[ERROR] No se pudo leer el frame de la cámara. La cámara se ha desconectado o ha fallado.")
                self.is_camera_connected = False # Marcar como desconectada
                CAMERA_ACTIVE = False
                # Si la grabación está activa, liberarla
                if VIDEO_WRITER is not None:
                    VIDEO_WRITER.release()
                    VIDEO_WRITER = None
                    SAVE_VIDEO = False
                self.cap.release() # Liberar recursos de la cámara
                continue # Ir al inicio del bucle para intentar reconectar o servir placeholder

            # Procesar el frame con el modelo de detección (solo si la cámara está activa y se leyó un frame)
            processed_frame, frame_detections = detection_service.process_frame(frame.copy(), update_global_counts=True)
            
            # Lógica de grabación de video
            # Inicia grabación si hay CUALQUIER detección y no se está grabando
            if any(count > 0 for class_name, count in frame_detections.items()) and not SAVE_VIDEO:
                SAVE_VIDEO = True
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs(UPLOAD_VIDEOS_FOLDER, exist_ok=True)
                OUTPUT_VIDEO_PATH = os.path.join(UPLOAD_VIDEOS_FOLDER, f"detection_{timestamp}.mp4")
                
                # Asegurarse de usar las dimensiones y FPS correctos para el grabador
                # Si la cámara cambia de resolución o FPS durante la ejecución, esto se actualizará al reconectar
                VIDEO_WRITER = cv2.VideoWriter(OUTPUT_VIDEO_PATH, self.video_fourcc, self.video_fps, self.video_dims)
                
                RECORDING_STARTED_AT = datetime.datetime.now()
                print(f"[INFO] Iniciando grabación de video en: {OUTPUT_VIDEO_PATH}")

            # Escribir el frame en el archivo de video si la grabación está activa
            if SAVE_VIDEO and VIDEO_WRITER is not None:
                VIDEO_WRITER.write(processed_frame)
                # Detener grabación después de 30 segundos (ejemplo de duración)
                if RECORDING_STARTED_AT and (datetime.datetime.now() - RECORDING_STARTED_AT).total_seconds() > 30:
                    SAVE_VIDEO = False
                    VIDEO_WRITER.release()
                    VIDEO_WRITER = None
                    print(f"[INFO] Grabación finalizada en: {OUTPUT_VIDEO_PATH}")
            
            # Codificar el frame procesado para la transmisión al navegador
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

camera_stream_service = CameraStream()

# --- Creación de la Aplicación Flask ---
app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

app.config['UPLOAD_IMAGES_FOLDER'] = UPLOAD_IMAGES_FOLDER
app.config['UPLOAD_VIDEOS_FOLDER'] = UPLOAD_VIDEOS_FOLDER

# --- Rutas de la Aplicación Flask ---

@app.route('/')
def index():
    current_counts = camera_stream_service.get_counts()
    return render_template('index.html', counts=current_counts, camera_active=CAMERA_ACTIVE)

@app.route('/en')
def index_en():
    current_counts = camera_stream_service.get_counts()
    return render_template('index-en.html', counts=current_counts, camera_active=CAMERA_ACTIVE)

@app.route('/video')
def video():
    """Ruta para el stream de video de la cámara."""
    return Response(camera_stream_service.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/counts')
def counts():
    """Ruta para obtener los conteos de objetos detectados."""
    return jsonify(camera_stream_service.get_counts())

@app.route('/upload', methods=['POST'])
def upload():
    """Ruta para subir y procesar imágenes estáticas."""
    if 'image' not in request.files:
        print("[ERROR] No se ha subido ninguna imagen en la petición /upload.")
        return jsonify({"error": "No se ha subido ninguna imagen"}), 400
    
    file = request.files['image']
    if file.filename == '':
        print("[ERROR] Nombre de archivo vacío en la petición /upload.")
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename_base, file_extension = os.path.splitext(file.filename)
    unique_filename = f"{filename_base}_{timestamp}{file_extension}"
    filepath = os.path.join(app.config['UPLOAD_IMAGES_FOLDER'], unique_filename)
    file.save(filepath)
    print(f"[INFO] Imagen subida guardada en: {filepath}")

    image = cv2.imread(filepath)
    if image is None:
        print(f"[ERROR] No se pudo cargar la imagen desde: {filepath}")
        return jsonify({"error": "No se pudo cargar la imagen"}), 500

    # Procesar la imagen subida, sin actualizar los conteos globales del stream
    processed_image, image_specific_counts = detection_service.process_frame(image.copy(), update_global_counts=False)
    
    # Aquí puedes decidir si los conteos de la imagen subida se suman a los globales
    # Actualmente, sí se suman. Si solo quieres que sean independientes, quita el siguiente bloque.
    with GLOBAL_COUNTS_LOCK:
        for class_name, count in image_specific_counts.items():
            GLOBAL_CLASS_COUNTS[class_name] += count

    output_filename = f"result_{unique_filename}"
    output_path = os.path.join(app.config['UPLOAD_IMAGES_FOLDER'], output_filename)
    cv2.imwrite(output_path, processed_image)
    print(f"[INFO] Imagen procesada guardada en: {output_path}")

    image_url = f"/static/uploads/images/{output_filename}"

    print(f"[DEBUG_UPLOAD] Imagen procesada URL: {image_url}. Conteos para gráfica: {image_specific_counts}")
    
    return jsonify({"image_url": image_url, "image_counts": image_specific_counts})

# --- Ejecución de la Aplicación ---
if __name__ == '__main__':
    # Asegúrate de que las carpetas de subida existan antes de iniciar la app
    os.makedirs(UPLOAD_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(UPLOAD_VIDEOS_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'static', 'images'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'static', 'logos'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'static', 'css'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'static', 'flags'), exist_ok=True)

    app.run(debug=True, threaded=True)