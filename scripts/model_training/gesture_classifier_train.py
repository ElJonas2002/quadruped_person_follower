import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow import keras

LANDMAKRS_DATASET_PATH = "datasets/HaGRID_120k/landmarks.csv"
LANDMARK_COUNT = 21
WRIST_IDX = 0           # muñeca (referencia de posición y de z)
MIDDLE_MCP_IDX = 9      # base del dedo medio (referencia de tamaño)
 
# -----------------------------------------------------------------------------
# Utilidades de forma
# -----------------------------------------------------------------------------
def to_landmarks(flat):
    """(N, 63) -> (N, 21, 3)"""
    return np.asarray(flat, dtype=np.float32).reshape(-1, LANDMARK_COUNT, 3)
 
 
def to_flat(landmarks):
    """(N, 21, 3) -> (N, 63)"""
    return landmarks.reshape(-1, LANDMARK_COUNT * 3)


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


def load_csv(csv_path):
    """
    Cargar CSV con los landmarks extraídos mediante MediaPipeHands junto con la clase y el split al que pertenece cada feature

    Returns:
        X: Datos de entrada (train y val)
        Y: Datos de las clases esperada (train y val)
    """
    # Leer dataset de landmarks de la mano
    df = pd.read_csv(csv_path)
    print(df.head())

    # Extraer features de la mano (63 features)
    features = [c for c in df.columns if c not in ('clase')]

    # Extraer datos de entrada, salida y splits
    X = df[features].values.astype(np.float32)
    Y = df['clase'].values

    return X, Y


def prepare_dataset(csv_path, show_splits=True):
    """
    Prepara los datos de entrenamiento (codifica las clases de salida y normaliza los datos de entrada) y crea los splits aleatoriamente

    Returns:
        X_train: Split de datos de entrada para entrenamiento
        Y_train: Split de clases de salida codificadas para entrenamiento
        X_val: Split de datos de entrada para validación
        Y_val: Split de clases de salida codificadas para validación
        le: Clases de salida codificadas
    """
    X, Y = load_csv(csv_path)

    # Codificar clase
    le = LabelEncoder()

    y_idx = le.fit_transform(Y)
    num_classes = len(le.classes_)

    X = normalize(X)

    # Split aleatorio estratificado
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, y_idx, test_size=0.2, stratify=y_idx, random_state=42
    )

    # Comprobar datos de cada split
    print(f"Clases: {list(le.classes_)}  (n={num_classes})")
    print(f"Train: {X_train.shape} | Val: {X_val.shape}")


    # --- Imprimir distribución por clase y split ---
    if show_splits:
        classes = list(le.classes_)
        train_counts = np.array([np.sum(Y_train == i) for i in range(num_classes)])
        val_counts   = np.array([np.sum(Y_val   == i) for i in range(num_classes)])
        totals       = train_counts + val_counts

        # Ancho dinámico para alinear nombres de clase
        name_width = max(len(c) for c in classes)
        bar_width  = 30  # caracteres para la barra

        print("\n" + "=" * 70)
        print("  Distribución del dataset por clase y split")
        print("=" * 70)
        print(f"  {'Clase'.ljust(name_width)}  {'Train':>7}  {'Val':>5}  {'Total':>6}  {'% Train':>8}  {'% Val':>6}")
        print("  " + "-" * (name_width + 50))

        for i, cls in enumerate(classes):
            tr, vl, tot = train_counts[i], val_counts[i], totals[i]
            pct_tr = tr / tot * 100
            pct_vl = vl / tot * 100
            print(f"  {cls.ljust(name_width)}  {tr:>7}  {vl:>5}  {tot:>6}  {pct_tr:>7.1f}%  {pct_vl:>5.1f}%")

        print("  " + "-" * (name_width + 50))
        total_train = train_counts.sum()
        total_val   = val_counts.sum()
        total_all   = totals.sum()
        print(f"  {'TOTAL'.ljust(name_width)}  {total_train:>7}  {total_val:>5}  {total_all:>6}  "
              f"{total_train/total_all*100:>7.1f}%  {total_val/total_all*100:>5.1f}%")

    return X_train, Y_train, X_val, Y_val, le

# -----------------------------------------------------------------------------
# Transformaciones de augmentación
# -----------------------------------------------------------------------------
def rotate_xy(landmarks, max_deg=25.0, rng=None):
    """Rota cada mano un ángulo aleatorio alrededor del eje z (rotación en plano)."""
    rng = rng or np.random
    angles = rng.uniform(-max_deg, max_deg, size=landmarks.shape[0]) * np.pi / 180.0
    cos, sin = np.cos(angles), np.sin(angles)
    out = landmarks.copy()
    x, y = out[..., 0].copy(), out[..., 1].copy()
    out[..., 0] = x * cos[:, None] - y * sin[:, None]
    out[..., 1] = x * sin[:, None] + y * cos[:, None]
    return out
 
 
def scale_jitter(landmarks, min_scale=0.9, max_scale=1.1, rng=None):
    """Escala aleatoria por mano. Tras la normalización, captura variaciones residuales."""
    rng = rng or np.random
    factors = rng.uniform(min_scale, max_scale, size=(landmarks.shape[0], 1, 1)).astype(np.float32)
    return landmarks * factors
 
 
def gaussian_noise(landmarks, sigma=0.02, rng=None):
    """Ruido gaussiano por landmark. Es el proxy de 'mediapipe inestable a distancia'."""
    rng = rng or np.random
    noise = rng.normal(0.0, sigma, size=landmarks.shape).astype(np.float32)
    return landmarks + noise
 
 
def horizontal_flip(landmarks, prob=0.5, rng=None):
    """Espejo horizontal (mano izquierda <-> derecha). OK para gestos simétricos."""
    rng = rng or np.random
    mask = rng.random(landmarks.shape[0]) < prob
    out = landmarks.copy()
    out[mask, :, 0] = -out[mask, :, 0]
    return out
 
# Pipeline de augmentación de datos
def augment(landmarks_flat, sigma=0.02, max_rot_deg=25.0,
            scale_range=(0.9, 1.1), flip_prob=0.5, rng=None):
    """
    Aplica el pipeline completo. Entrada y salida: (N, 63) normalizadas.
 
    Orden: rotación -> escala -> flip -> ruido.
    El ruido va al final para que represente el jitter de mediapipe sobre lo
    que sea que ya tengamos.
    """
    lm = to_landmarks(landmarks_flat)
    lm = rotate_xy(lm, max_rot_deg, rng=rng)
    lm = scale_jitter(lm, *scale_range, rng=rng)
    lm = horizontal_flip(lm, flip_prob, rng=rng)
    lm = gaussian_noise(lm, sigma, rng=rng)
    return to_flat(lm)

# Construir modelo de Keras
def build_model(input_size=63, output_size=4):
    """
    Construye el modelo del clasificador de gestos en Keras
    Args:
        input_size:     Tamaño del vector de entradas
        output_size:    Tamaño de la capa de salida
    Returns:
        model: Clasificador de Keras con 2 capas ocultas (64 y 32 neuronas) y función de activación ReLu.
    """
    model = keras.Sequential()
    model.add(keras.Input(shape = (input_size,))),     # Especificar tamaño de la capa de entrada
    model.add(keras.layers.Dense(64, activation = "relu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(32, activation = "relu"))
    model.add(keras.layers.Dense(output_size, activation = "softmax"))   # Número de clases como neuronas de salida. Softmax para One vs All

    return model

# Graficar métricas de desempeño
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  plt.figure(figsize = (15,5))
  plt.subplot(131)
  plt.title('Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.plot(hist['epoch'], hist['loss'],
           label='Train')
  plt.plot(hist['epoch'], hist['val_loss'],
           label = 'Val')
  plt.yscale('log')
  plt.legend()

  plt.subplot(132)
  plt.title('F1-score')
  plt.xlabel('Epoch')
  plt.ylabel('F1-score')
  plt.plot(hist['epoch'], hist['f1_macro'],
           label='Train')
  plt.plot(hist['epoch'], hist['val_f1_macro'],
           label = 'Val')
  plt.legend()

  plt.subplot(133)
  plt.title('Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(hist['epoch'], hist['accuracy'],
           label='Train')
  plt.plot(hist['epoch'], hist['val_accuracy'],
           label = 'Val')
  
  plt.savefig("models/gesture_classifier/reports/metric_plots_v4.png")

# Obtener F1Score en cada época
class F1ScoreCallback(keras.callbacks.Callback):
    """Calcula F1-score macro sobre train y val al final de cada época."""

    def __init__(self, X_train, y_train, X_val, y_val, average='macro'):
        super().__init__()
        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val
        self.average = average

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}

        # Predicciones (argmax sobre la salida softmax)
        y_pred_train = np.argmax(self.model.predict(self.X_train, verbose=0), axis=1)
        y_pred_val   = np.argmax(self.model.predict(self.X_val,   verbose=0), axis=1)

        # F1-score con sklearn
        f1_train = f1_score(self.y_train, y_pred_train, average=self.average)
        f1_val   = f1_score(self.y_val,   y_pred_val,   average=self.average)

        # Registrar en logs para que aparezcan en history
        logs['f1_macro']     = f1_train
        logs['val_f1_macro'] = f1_val

        print(f" — f1_macro: {f1_train:.4f} — val_f1_macro: {f1_val:.4f}")

# Generador con augmentación on-the-fly
class AugmentedSequence(keras.utils.Sequence):
    """
    Genera batches con augmentación aplicada cada época.
    Cada época pasa por todas las muestras una vez, pero augmentadas distinto.
    """
    def __init__(self, X, y, batch_size=128, shuffle=True,
                 sigma=0.025, max_rot_deg=25.0,
                 scale_range=(0.9, 1.1), flip_prob=0.5, seed=0):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sigma = sigma
        self.max_rot_deg = max_rot_deg
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(X))
        self.on_epoch_end()
 
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
 
    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        Xb = self.X[batch_idx]
        yb = self.y[batch_idx]
        Xb_aug = augment(Xb,
                         sigma=self.sigma,
                         max_rot_deg=self.max_rot_deg,
                         scale_range=self.scale_range,
                         flip_prob=self.flip_prob,
                         rng=self.rng)
        return Xb_aug, yb
 
    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


if __name__=="__main__":

    # --- Preparar dataset ---
    x_train, y_train, x_val, y_val, le = prepare_dataset(LANDMAKRS_DATASET_PATH)
    num_classes = len(le.classes_)

    # --- Crear augmentation para entrenamineto ---
    train_gen = AugmentedSequence(x_train, y_train, batch_size=128,
                                sigma=0.015, 
                                max_rot_deg=10,
                                scale_range=(0.9, 1.1), 
                                flip_prob=0.0)
    
    val_data = (x_val, y_val)

    f1_cb = F1ScoreCallback(x_train, y_train, x_val, y_val)

    # --- Construir y compilar modelo ---
    model = build_model(input_size=x_train.shape[1], 
                        output_size=len(le.classes_))
    
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), 
                optimizer=keras.optimizers.Adam(),
                metrics=["accuracy"])

    model.summary()

    # --- Callbacks de entrenamiento (regularizaciones y f1-macro) ---
    callbacks = [
        # Early Stopping
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        # Reducir Learning Rate en estancamiento
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5
        ),
        f1_cb
    ]

    # --- Entrenar modelo y almacenar historial ---
    history = model.fit(
        train_gen,
        validation_data = val_data,
        epochs = 200,
        callbacks = callbacks,
        verbose = 2
    )

    # --- Graficar métricas (loss, acc, f1-macro) ---
    plot_history(history)

    # --- Reporte de clasificación ---
    y_pred = np.argmax(model.predict(x_val), axis=1)
    report = classification_report(y_val, y_pred, target_names=le.classes_)
    print(report)

    # --- Guardar reporte en txt ---
    with open("models/gesture_classifier/reports/classification_report_v4.txt", "w") as f:
        f.write(report)

    # --- Matriz de confusión ---
    cm = confusion_matrix(y_val, y_pred)

    # --- Visualizar y guardar como imagen ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.ylabel("Real value")
    plt.xlabel("Predicted value")
    plt.tight_layout()
    plt.savefig("models/gesture_classifier/reports/confusion_matrix_v4.png", dpi=150)
    plt.show()

    print("✓ Reporte guardado en reports/classification_report.txt")
    print("✓ Matriz guardada en reports/confusion_matrix.png")

    weights = model.get_weights()
    np.savez("models/gesture_classifier/gesture_weights_v4.npz", *weights)
    print("✓ Pesos guardados como numpy")

    # # Descomentar para exportar modelo a TFLite (portable)
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()

    # with open("models/gesture_classifier/gesture_classifier.tflite", "wb") as f:
    #     f.write(tflite_model)
    # print("✓ gesture_classifier.tflite guardado en models/gesture_classifier")

    # --- Guardar orden de clases para inferencia ---
    with open("models/gesture_classifier/gesture_classes.json", "w") as f:
        json.dump(le.classes_.tolist(), f)

    print("✓ gesture_classes.json guardado en models/gesture_classifier")