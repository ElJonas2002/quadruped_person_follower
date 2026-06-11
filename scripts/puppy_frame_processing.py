import cv2
import numpy as np
import time
import mediapipe as mp
from tensorflow import keras
from collections import deque

WRIST_IDX = 0                   # índice de la muñeca (referencia de posición y de z)
MIDDLE_MCP_IDX = 9              # índice de la base del dedo medio (referencia de tamaño)

def to_landmarks(flat):
    """(N, 63) -> (N, 21, 3)"""
    return np.asarray(flat, dtype=np.float32).reshape(-1, 21, 3)
 
 
def to_flat(landmarks):
    """(N, 21, 3) -> (N, 63)"""
    return landmarks.reshape(-1, 21 * 3)


def normalize(landmarks):
    """
    landmarks: (N, 21, 3) o (N, 63)
    Devuelve landmarks centrados en la muñeca y escalados de forma que la
    distancia muñeca-MCP-medio sea 1. Resultado en la misma forma de entrada.
    """
    flat_input = landmarks.ndim == 2 and landmarks.shape[-1] == 63
    lm = to_landmarks(landmarks) if flat_input else landmarks.astype(np.float32).copy()
 
    # --- Centrar en la muñeca ---
    wrist = lm[:, WRIST_IDX:WRIST_IDX + 1, :]      # (N, 1, 3)
    lm = lm - wrist
 
    # --- Escalar por distancia muñeca-MCP-medio ---
    ref = lm[:, MIDDLE_MCP_IDX, :]                  # (N, 3)
    scale = np.linalg.norm(ref, axis=1, keepdims=True)  # (N, 1)
    scale = np.where(scale > 1e-6, scale, 1.0)
    lm = lm / scale[:, :, None]

    return to_flat(lm) if flat_input else lm


def load_params(npz_path):
    """
    Carga los parámetros de calibración de la cámara desde un archivo .npz.
    """
    data = np.load(str(npz_path))
    params = {
        "K": data["K"],
        "dist": data["dist"],
        "img_size": tuple(data["img_size"].tolist()),
        "rms_error": float(data["rms_error"]),
    }
    if "square_size_mm" in data:
        params["square_size_mm"] = float(data["square_size_mm"])

    return params

def build_undistort_maps(K, dist, img_size):
    """
    Precomputa los mapas de corrección de distorsión. Llamar una sola vez al inicio.
    Usar con undistort_image() para corregir frames a alta velocidad con cv2.remap.

    Args:
        K: Matriz intrínseca calibrada
        dist: Arreglo numpy de coeficientes de distorsión
        img_size: Tamaño de la imagen (w, h)
    Returns:
        map1, map2: Mapas de remapeo para cv2.remap
        roi_slice: Slice numpy para recortar bordes negros (o None)
    """
    w, h = img_size
    K_optimal, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K_optimal, (w, h), cv2.CV_16SC2)
    x, y, rw, rh = roi
    roi_slice = (slice(y, y + rh), slice(x, x + rw)) if rw > 0 and rh > 0 else None
    return map1, map2, roi_slice


def undistort_image(img, map1, map2, roi_slice=None):
    """
    Aplica corrección de distorsión usando mapas precomputados (cv2.remap).
    Mucho más rápido que cv2.undistort por frame — el costo de getOptimalNewCameraMatrix
    se paga solo una vez en build_undistort_maps.

    Args:
        img: Frame BGR de 8 bits a corregir.
        map1, map2: Mapas obtenidos de build_undistort_maps.
        roi_slice: Slice de recorte obtenido de build_undistort_maps.
    """
    undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    return undistorted[roi_slice] if roi_slice is not None else undistorted


def build_gesture_model(weights_path=None, input_size=63, num_classes=4):
    """
    Construye el modelo de Keras del clasificador de gestos
    Args:
        input_size: Tamaño del vector de entradas
        num_classes: Número de clases (salidas) del problema
    """
    model = keras.Sequential([
        keras.Input(shape=(input_size,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    if weights_path:
        data = np.load(weights_path)
        model.set_weights([data[k] for k in sorted(data.files)])
    return model


def extract_landmarks(hand_detector, img=None, video=True, start_time=None, min_handedness_score=0.7):
    """
    Extrae 63 landmarks XYZ de la primera mano visible en la imagen (o frame).
    Args:
        hand_detector: Modelo de detección de keypoints de MediaPipe Hands.
        num_hands: Número de manos a detectar
        img: Imagen (o frame) de OpenCV en BGR.
        video: Habilitar procesamiento de video (True) o imagen (False).
        start_time: Tiempo inicial de procesamiento (solo para video).
        min_handedness_score: Score mínimo de handedness para aceptar la detección.
    Returns:
        coords: Lista de coordenadas (x,y,z) de los 21 keypoints (63 valores)
    """

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    if video:
        timestamp = int((time.time() - start_time) * 1000)
        result = hand_detector.detect_for_video(mp_img, timestamp)
    else:
        result = hand_detector.detect(mp_img)

    if not result.hand_landmarks:
        return None

    # Rechazar detecciones donde MediaPipe no está seguro de que sea una mano
    if result.handedness and result.handedness[0][0].score < min_handedness_score:
        return None

    lms = result.hand_landmarks[0]
    arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)  # (21, 3)

    return arr.ravel().tolist()   # (63, 1)


def draw_landmarks(img, hand_coords):
    """
    Dibujar los landmarks reales de la mano detectada en el frame (para visualización)

    Args:
        img: Imagen con la mano detectada
        hand_coords: Lista con las coordenadas de cada eje [[X],[Y]]
    """
    coords_xyz = np.array(hand_coords, dtype=np.float32).reshape(21, 3)
    xs, ys  = coords_xyz[:,0], coords_xyz[:,1]
    for px, py in zip((xs), (ys)):
        cv2.circle(img, (int(px), int(py)), 3, (255, 0, 0), -1)
    
    return img


def track_person(yolo_model, frame):
    """
    Realiza la detección (localización + seguimiento) de las personas encontradas en el frame utilizando YOLO+BoTSORT

    Args:
        yolo_model: Modelo de YOLOv8 entrenado para detectar personas
        frame: Frame en el cual se detectarán las personas

    Returns:
        persons: Lista de diccionarios con las personas detectadas en el frame: {"ID", "bbox"}
    """
    results = yolo_model.track(frame, classes=[0], persist=True, verbose=False)
    if results[0].boxes.id is None:
        return None

    person_id = int(results[0].boxes[0].id.item())
    person_bbox = list(map(int, results[0].boxes[0].xyxy[0].tolist()))

    return {"id": person_id, "bbox": person_bbox}


def crop_person(img, person_bbox, padding_ratio=0.15):
    """
    Recortar persona detectada en el frame para detectar landmarks de la mano a distancia con MediaPipe Hands

    Args:
        img: Imagen con la persona detectada
        person_bbox: Bounding box de la persona detectada (x1,y1,x2,y2)
        padding_ratio: Razón del padding del bbox para evitar cortar la mano
    
    Returns:
        person_crop: Imagen recortada de la persona detectada + padding aplicado a su bbox
        new_bbox: Lista de puntos (x1,y1,x2,y2) del bbox con padding de la persona detectada
    """
    x1, y1, x2, y2 = person_bbox
    h, w = img.shape[:2]
    dw, dh = x2 - x1, y2 - y1

    pad_x = int(dw * padding_ratio)
    pad_y = int(dh * padding_ratio)

    new_x1 = max(0, x1 - pad_x)
    new_x2 = min(w, x2 + pad_x)
    new_y1 = max(0, y1 - pad_y)
    new_y2 = min(h, y2 + pad_y)

    person_crop = img[new_y1:new_y2, new_x1:new_x2]
    new_person_bbox = [new_x1, new_y1, new_x2, new_y2]

    return person_crop, new_person_bbox


def map_landmarks(hand_coords, person_bbox):
    """
    Mapea los landmarks normalizados --obtenidos en el recorte del frame con el bbox de la persona-- a su posición real en el frame original

    Args:
        hand_coords: Coordenadas de la mano normalizadas obtenidas de MediaPipe en el frame recortado
        person_bbox: Coordenadas (x1,y1,x2,y2) del bbox utilizado para recortar el frame original.
    Returns:
        real_coords: Coordenadas de la mano detectada mapeadas con respecto al frame original
    """
    x1, y1, x2, y2 = person_bbox
    new_w, new_h = x2 - x1, y2 - y1

    coords_xyz = np.array(hand_coords, dtype=np.float32).reshape(21, 3)

    coords_xyz[:,0] = coords_xyz[:,0] * new_w + x1
    coords_xyz[:,1] = coords_xyz[:,1] * new_h + y1

    return coords_xyz.ravel().tolist()  # (63,1)


class ClassifyGesture:
    def __init__(self, classes, window=5, min_conf=0.6, min_agree=3, val_rate=0.75):
        self.classes = classes
        self.gestures_queue = deque(maxlen=window)
        self.confs_queue = deque(maxlen=window)
        self.min_conf = min_conf
        self.min_agree = min_agree
        self.val_rate = val_rate

    def reset_window(self):
        self.gestures_queue.clear()
        self.confs_queue.clear()
    
    def update_window(self, model, hand_coords):
        # --- 1. Clasificar gesto que aparece en el frame ---
        input = np.array(hand_coords, dtype=np.float32).reshape(1, -1)
        norm_input = normalize(input)
        #output = model.predict(norm_input, verbose=0)[0]
        output = model(norm_input, training=False).numpy()[0]
        idx = np.argmax(output)
        conf = output[idx]

        # --- 2. Añadirlo a la ventana desliante si supera cierta confianza ---
        self.gestures_queue.append(idx if conf >= self.min_conf else -1)
        self.confs_queue.append(conf if conf >= self.min_conf else -1)

        # --- 3. Seleccionar gestos válidos de la ventana ---
        valid_frames = [i for i in self.gestures_queue if i >= 0]

        # --- 4. Generar arreglo de frecuencias de ocurrencia de cada gesto y validar ---
        if len(valid_frames) < self.min_agree:
            return None, 0.0
        
        counts = np.bincount(valid_frames, minlength=len(self.classes))

        if counts.max()/len(valid_frames) < self.val_rate:             # val_rate% de los gestos para asegurar
            return None, 0.0
    
        stable_idx = int(counts.argmax())
        gesture = self.classes[stable_idx]
        conf = np.mean([c for p, c in zip(self.gestures_queue, self.confs_queue) if p == stable_idx])

        return gesture, conf