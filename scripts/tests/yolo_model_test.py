import cv2
import time
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PERSON_MODEL_PATH = "models/person_detector/person_v3_original_data_medium/weights/best.engine"
HAND_MODEL_PATH   = "models/hand_landmarker.task"

def extract_landmarks(hand_detector, img=None, video=True, start_time=None, draw_points=False):
    """
    Extrae 63 landmarks XYZ de la primera mano visible en la imagen (o frame).
    Args:
        hand_detector: Modelo de detección de keypoints de MediaPipe Hands.
        img: Imagen (o frame) de OpenCV en BGR.
        video: Habilitar procesamiento de video (True) o imagen (False).
        start_time: Tiempo inicial de procesamiento (solo para video).
        draw_points: Dibujar puntos de la mano obtenidos.
    Returns:
        img_rgb: Imagen RGB con los keypoints dibujados
        coords: Arreglo de coordenadas (x,y,z) de los keypoints de la mano
        bbox_norm: Coordenadas (x1, y1, x2, y2) normalizadas del bbox de la mano
    """

    result = None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    if video:
        timestamp = int((time.time() - start_time) * 1000)
        result = hand_detector.detect_for_video(mp_img, timestamp)
    else:
        result = hand_detector.detect(mp_img)

    if not result.hand_landmarks:
        return img, None, None

    xs, ys, coords = [], [], []
    for lm in result.hand_landmarks[0]:
        x, y, z = lm.x, lm.y, lm.z
        coords.extend([x, y, z])
        xs.append(x)
        ys.append(y)

        # -- Dibujar puntos reales --
        if draw_points:
            x = int(x * img.shape[1])
            y = int(y * img.shape[0])
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

    bbox_norm = [min(xs), min(ys), max(xs), max(ys)]

    return img, coords, bbox_norm

# --- Importar modelo de YOLOv8 para detección ---
model = YOLO(PERSON_MODEL_PATH, task="detect")

# --- Crear objeto detector de MediaPipe Hands ---
base_opts = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
mp_opts = vision.HandLandmarkerOptions(
    base_options=base_opts,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

hand_detector = vision.HandLandmarker.create_from_options(mp_opts)

src = cv2.VideoCapture(0)

if not src.isOpened():
    print("No se pudo abrir la cámara")
    exit()

start_time = time.time()

while(True):
    ret, frame = src.read()

    if not ret:
        print("No se pudo obtener el frame")
        break

    frame, _, _ = extract_landmarks(hand_detector, frame, start_time=start_time, draw_points=True)
    results = model.track(frame, classes=[0], persist=True, verbose=False)
    frame = results[0].plot()

    cv2.imshow("Modelo YOLO", frame)

    if cv2.waitKey(1) & 0xFF==27:
        break

cv2.destroyAllWindows()
