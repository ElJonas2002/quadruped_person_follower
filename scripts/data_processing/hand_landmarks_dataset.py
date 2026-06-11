import cv2
import json
import os
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Configuración ──────────────────────────────────────────────
MODEL_PATH   = "models/hand_landmarker.task"
DATASET_DIR  = "dataset/HaGRID_120k/hagrid_120k"
JSON_DIR    = "dataset/HaGRID_120k"
OUTPUT_CSV   = "dataset/HaGRID_120k/landmarks.csv"
VAL_RATIO    = 0.2  # 20% para validación

GESTOS = {
    "fist" : "Continue",
    "one"  : "Follow_Me",
    "peace" : "Hi!",
    "palm" : "Stop",
}
# ──────────────────────────────────────────────────────────────

# Inicializar MediaPipe en modo IMAGE
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

def recortar_mano(imagen, bbox, margen=0.1):
    """Recorta la mano usando el bbox normalizado del JSON con un margen extra."""
    h, w = imagen.shape[:2]
    cx, cy, bw, bh = bbox

    # Convertir a píxeles y agregar margen
    x1 = int((cx - bw/2 - margen) * w)
    y1 = int((cy - bh/2 - margen) * h)
    x2 = int((cx + bw/2 + margen) * w)
    y2 = int((cy + bh/2 + margen) * h)

    # Clampear a los límites de la imagen
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    return imagen[y1:y2, x1:x2]

def extraer_landmarks(roi):
    """Extrae 21 landmarks XYZ normalizados del ROI."""
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]

    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y, lm.z])  # 21 puntos × 3 = 63 features

    return coords

def procesar_gesto(nombre_gesto, clase_destino, escritor_csv, contadores):
    carpeta_imagenes = os.path.join(DATASET_DIR, f"train_val_{nombre_gesto}")
    json_path        = os.path.join(JSON_DIR, "ann_train_val", f"{nombre_gesto}.json")

    if not os.path.exists(carpeta_imagenes):
        print(f"⚠️  No encontrada: {carpeta_imagenes}")
        return
    if not os.path.exists(json_path):
        print(f"⚠️  No encontrado: {json_path}")
        return

    with open(json_path, "r") as f:
        anotaciones = json.load(f)

    ids = list(anotaciones.keys())
    total = len(ids)
    n_val = int(total * VAL_RATIO)

    print(f"\n  Procesando '{nombre_gesto}' → '{clase_destino}' ({total} imágenes)")

    for i, img_id in enumerate(ids):
        datos = anotaciones[img_id]

        # Buscar la imagen (puede ser .jpg o .png)
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            ruta = os.path.join(carpeta_imagenes, img_id + ext)
            if os.path.exists(ruta):
                img_path = ruta
                break

        if img_path is None:
            continue

        imagen = cv2.imread(img_path)
        if imagen is None:
            continue

        bbox = datos["bboxes"][0]
        roi  = recortar_mano(imagen, bbox)

        if roi.size == 0:
            continue

        landmarks = extraer_landmarks(roi)
        if landmarks is None:
            contadores["sin_landmarks"] += 1
            continue

        split = "val" if i < n_val else "train"
        fila  = landmarks + [clase_destino, split]
        escritor_csv.writerow(fila)
        contadores["exitosas"] += 1

        # Progreso cada 100 imágenes
        if contadores["exitosas"] % 100 == 0:
            print(f"    {contadores['exitosas']} landmarks extraídos...")

def main():
    contadores = {"exitosas": 0, "sin_landmarks": 0}

    # Encabezado del CSV: x0,y0,x1,y1,...,x20,y20,clase,split
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header += ["clase", "split"]

    with open(OUTPUT_CSV, "w", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(header)

        print("\n╔══════════════════════════════════════════╗")
        print("║  Extracción de landmarks - HaGRID        ║")
        print("╚══════════════════════════════════════════╝")

        for gesto, clase in GESTOS.items():
            procesar_gesto(gesto, clase, escritor, contadores)

    print(f"\n╔══════════════════════════════════════════╗")
    print(f"║  Resumen                                 ║")
    print(f"╚══════════════════════════════════════════╝")
    print(f"  ✓ Landmarks extraídos: {contadores['exitosas']}")
    print(f"  ✗ Sin detección:       {contadores['sin_landmarks']}")
    print(f"  Guardado en:           {OUTPUT_CSV}")

if __name__ == "__main__":
    main()