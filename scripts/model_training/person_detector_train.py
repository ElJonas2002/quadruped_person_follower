from ultralytics import YOLO

MODELS_PATH = "/home/jonas02/ml_project/models/person_detector"
NEW_MODEL_NAME = "v4_cleanned_data_small"

# Descargar modelo preentrenado en COCO ó reanudar entrenamiento apuntando a archivo last.pt del modelo
model = YOLO("/home/jonas02/ml_project/models/person_detector/v4_cleanned_data_small/weights/last.pt")  

model.train(
    data = "/home/jonas02/ml_project/datasets/person.yaml",
    project = MODELS_PATH,
    name = NEW_MODEL_NAME,
    verbose = True,
    exist_ok = True,
    amp = False,
    resume = True,          # Reanudar entrenamiento usando last.pt
    epochs = 200,
    imgsz = 640,            # Reescalado de imágenes
    batch = 16,             # Entrenamiento por lote
    workers = 8,            # Hilos de CPU (default value)
    device = 0,             # 0 -> GPU para entrenar
    optimizer = "AdamW",    # Optimizador
    lr0 = 0.001,            # LR inicial

    # Regularizaciones
    patience = 10,          # Early Stopping

    # Data augmentation
    augment = True,
    hsv_h = 0.015,          # Cambiar Hue aleatoriamente entre -x y x (0 es la imagen original)
    hsv_s = 0.7,            # Cambiar Saturation aleatoriamente entre -x y x (0 es la imagen original)
    hsv_v = 0.4,            # Cambiar Value aleatoriamente entre -x y x (0 es la imagen original)
    fliplr = 0.5,           # Virar horizontalmente con probabilidad de 50%
    scale = 0.5,            # Zoom aleatorio con factor de escalado de 50%
    translate = 0.25,       # Desplazar la imagen hasta 25% en x/y → persona sale cortada por bordes
    erasing = 0.4,          # Borrar aleatoriamente regiones de la imagen (simula oclusión parcial)
    perspective=0.0005,     # Distorsión de perspectiva leve, simula cámara inclinada 
    mosaic = 1.0,           # Combinar 4 imágenes en una con probabilidad de 50% para entrenar
)

# Validar modelo entrenado
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

# Exportar modelo en formato engine (evita instalar todo Pytorch) en CMD:
model = YOLO(f"{MODELS_PATH}/{NEW_MODEL_NAME}/weights/best.pt").export(format="engine", imgsz=640)