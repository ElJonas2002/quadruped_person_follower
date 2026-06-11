import cv2
import json
import numpy as np
import time
import mediapipe as mp
import tensorflow.lite as tflite
#from tensorflow import keras
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Configuración ──────────────────────────────────────────────
MODEL_PATH   = "models/gesture_classifier/gesture_classifier.tflite"
CLASSES_PATH = "models/gesture_classifier/gesture_classes.json"
HAND_MODEL   = "models/hand_landmarker.task"
CONFIANZA_MIN = 0.85
CAMARA_ID     = 0
# ──────────────────────────────────────────────────────────────

# Cargar clasificador TFLite
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Cargar modelo de Keras
# model = keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH) as f:
    clases = json.load(f)

# Inicializar MediaPipe en modo VIDEO
base_options = python.BaseOptions(model_asset_path=HAND_MODEL)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

def extraer_landmarks(frame, start_time):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp = int((time.time() - start_time) * 1000)
    result    = detector.detect_for_video(mp_image, timestamp)

    if not result.hand_landmarks:
        return None

    coords = []
    for lm in result.hand_landmarks[0]:
        coords.extend([lm.x, lm.y, lm.z])  # 63 features XYZ

    return np.array(coords, dtype=np.float32)

def clasificar_gesto(landmarks):
    entrada = np.array(landmarks, dtype=np.float32).reshape(1, -1)
    # Salida para tflite
    interpreter.set_tensor(input_details[0]['index'], entrada)
    interpreter.invoke()
    salida     = interpreter.get_tensor(output_details[0]['index'])[0]
    # Salia para keras
    #salida = model.predict(entrada, verbose=0)[0]
    confianza = float(np.max(salida))
    clase_idx = int(np.argmax(salida))
    return clases[clase_idx], confianza


if __name__ == "__main__":
    cap = cv2.VideoCapture(CAMARA_ID)
    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara")

    print("\n╔══════════════════════════════════════╗")
    print("║    Prueba clasificador de gestos     ║")
    print("╚══════════════════════════════════════╝")
    print(f"  Confianza mínima: {CONFIANZA_MIN}")
    print(f"  Clases: {clases}")
    print("\n  Presiona Q para salir\n")

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extraer_landmarks(frame, start_time)

        if landmarks is None:
            print("  [--] Sin mano detectada")
        else:
            gesto, confianza = clasificar_gesto(landmarks)

            if confianza >= CONFIANZA_MIN:
                print(f"  [✓] {gesto:<12} | confianza: {confianza:.2f}")
            else:
                print(f"  [?] {gesto:<12} | confianza: {confianza:.2f} (bajo umbral)")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()