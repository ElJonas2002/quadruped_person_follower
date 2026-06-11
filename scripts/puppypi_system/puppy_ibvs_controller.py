#!/usr/bin/env python3
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from sensor.msg import Led
from std_msgs.msg import String, Float32MultiArray, Empty as EmptyMsg
from std_srvs.srv import Empty
from puppy_control.msg import Velocity
from puppy_control.srv import SetRunActionName

import json
import time
from enum import Enum
import numpy as np

from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from puppy_frame_processing import *
from puppy_command import *

# --- Rutas relativas de modelos ---
HAND_MODEL_PATH = "models/hand_landmarker.task"
PERSON_MODEL_PATH = "models/person_detector/v4_cleanned_data_small/weights/best.onnx"
CLASSIFIER_WEIGHTS = "models/gesture_classifier/gesture_weights_v3.npz"
CLASSES_PATH = "models/gesture_classifier/gesture_classes.json"

# --- Parámetros ---
MIN_CONF = 0.80                 # Confianza mínima para aceptar un gesto

PERSON_H = 1.70                 # Altura estimada de una persona adulta (m)
TARGET_DIST = 3.0               # Distancia objetivo de seguimiento (m)

KP_LIN = 150                    # Ganancia proporcional — velocidad lineal
KP_ANG = 1.2                    # Ganancia proporcional — velocidad angular
MAX_VX = 15.0                   # Velocidad lineal máxima segura (cm/s)
MAX_VYAW = np.radians(20)       # Velocidad angular máxima segura (rad/s)
DB_X = 0.05                     # Banda muerta horizontal
DB_H = 0.1                      # Banda muerta de distancia
SLEW_RATE_LIN = 30              # Slew rate (cm/s²)
SLEW_RATE_ANG = np.radians(40)  # Slew rate (rad/s²)

TIMEOUT_HI = 3.0                # Duración del saludo (s)
FRAME_TOL = 10                  # Tolerancia de frames para que el objetivo vuelva

# --- Máquina de estados ---
class State(Enum):
    IDLE      = "inactivo"
    FOLLOWING = "siguiendo"
    STOPPED   = "pausado"
    HI        = "saludando"

# --- Limitador de la tasa de cambio de velocidades ---
class SlewRateLimiter:
    """Slew Rate Limiter: Evita cambios bruscos de velocidad al limitar su respectiva tasa de cambio (|du/dt| =< max_rate)"""
    def __init__(self, max_rate):
        """
        Args:
            max_rate: Maxima tasa de cambio de velocidad (cm/s² ó rad/s²)
        """
        self.max_delta = max_rate
        self.last_t = None
        self.last_vel = 0.0
    
    def update(self, vel):
        """
        Calcula la velocidad limitada por la tasa máxima de cambio
        Args:
            vel: Comando de velocidad lineal/angular
        """
        t = time.time()

        # --- Actualizar valores en la primera llamada (introduce un pequeño delay en la respuesta)
        if self.last_t is None:
            self.last_t = t
            self.last_vel = vel
            return 1e-6
        
        dt = t - self.last_t
        max_delta = self.max_delta * dt
        delta = vel - self.last_vel

        # --- Limitar tasa de cambio si se excede ---
        if abs(delta) > max_delta:
            delta = max_delta * (1 if delta > 0.0 else -1)
        
        self.last_vel += delta
        self.last_t = t
        return self.last_vel
    
    def reset(self):
        """Reiniciar variables para el cálculo del Slew Rate"""
        self.last_t = None
        self.last_vel = 0.0

def reset_slew_rates():
    sr_lin.reset()
    sr_ang.reset()    


# --- IBVS: Ley de control P ---
def compute_ibvs(bbox):
    x1, y1, x2, y2 = bbox
    w, h = img_size

    # --- Error horizontal normalizado: centroide de la persona vs. centro del frame ---
    cx_person = (x1 + x2) / 2
    error_x = (cx_person - cx_img) / w

    # --- Error de altura normalizado: altura del bbox deseada vs actual usando el modelo pinhole ---
    current_h = y2 - y1
    error_h = (TARGET_BBOX_H - current_h) / h       # Modelo pinhole: h_ratio = H_persona * fy / dist * FRAME_H

    # --- Ley de control P de velocidades con banda muerta y saturación ---
    if abs(error_x) < DB_X:
        vyaw_cmd = 0.0
    else:
        vyaw_cmd = -KP_ANG * error_x
        vyaw_cmd = max(-MAX_VYAW, min(MAX_VYAW, vyaw_cmd))

    if abs(error_h) < DB_H:
        vx_cmd = 0.0
    else:
        vx_cmd = KP_LIN * error_h
        vx_cmd = max(-MAX_VX, min(MAX_VX, vx_cmd))

    # --- Limitar posibles cambios bruscos con Slew Rate ---
    vx_cmd = sr_lin.update(vx_cmd)
    vyaw_cmd = sr_ang.update(vyaw_cmd)

    return vx_cmd, vyaw_cmd


# --- Función para acostar al robot al cerrar el nodo de ROS ---
def callback_shutdown(msg):
    color_state(pub=pub_rgb, color=colors["off"])
    stop(pub=pub_vel, client=go_home)
    run_action(client=action_group, action_name="lie_down")
    rospy.sleep(1)
    rospy.signal_shutdown("Cerrar Aplicación")


# --- Procesamiento de cada frame + máquina de estados ---
def callback_cam(msg):
    global state, target_id, hi_start, isInHome, action_executed, vyaw, frame_count
    gesture, conf = None, 0.0

    # --- Convertir frame a BGR de 8 bits para procesar con OpenCV ---
    frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
    frame = undistort_image(frame, map1, map2, roi_slice)

    person = track_person(yolo_model=yolo, frame=frame)

    # Publicar track solo si hay detección; dict vacío si no hay persona
    track_payload = {"id": person["id"], "bbox": person["bbox"], "is_target": person["id"] == target_id} if person else {}
    pub_person_track.publish(String(data=json.dumps(track_payload)))
    
    # --- Si hay persona, recorta el frame con el bbox de la persona y extrae landmarks de la mano detectada ---
    if person:
        person_crop, new_bbox = crop_person(frame, person["bbox"])
        coords = extract_landmarks(hand_detector, person_crop, start_time=start_time)

        # --- Si hay mano, publicar landmarks y clasificar gesto del frame recortado ---
        if coords is not None:
            mapped_coords = map_landmarks(coords, new_bbox)
            coords_msg = Float32MultiArray(data=mapped_coords)
            pub_hand_coords.publish(coords_msg)
        
            gesture, conf = gc.update_window(gesture_model, coords)
            pub_gesture.publish(String(data=f"{gesture}:{conf:.2f}:{state.value}"))

    # --- ESTADO HI: gira hacia el objetivo y lo saluda ---
    if state == State.HI:
        target = person if person and person["id"] == target_id else None
        
        if target and not action_executed:
            _, vyaw = compute_ibvs(target["bbox"])   # vx=0, solo girar
            publish_vel(pub_vel, 0.0, vyaw)
        
        if vyaw == 0.0:
            if not action_executed:
                hi_start = time.time()
                action_executed = run_action(client=action_group, action_name="shake_hands")

            if time.time() - hi_start >= TIMEOUT_HI:
                print("HI → IDLE (timeout)")
                state, target_id, action_executed = State.IDLE, None, False

        return

    # ── ESTADO IDLE: esperar Follow_Me o Hi! ──────────────────────────────────
    if state == State.IDLE:

        # Ir a home si no lo está
        if not isInHome:
            stop(pub=pub_vel, client=go_home)
            reset_slew_rates()
            color_state(pub=pub_rgb, color=colors["blue"])
            isInHome = True

        if gesture is None or person is None:
            return   # sin gesto válido/persona → seguir esperando

        if gesture == "Follow_Me":
            target_id = person["id"]   # única persona detectada
            state = State.FOLLOWING
            color_state(pub=pub_rgb, color=colors["green"])
            isInHome = False
            gc.reset_window()
            print(f"IDLE → FOLLOWING  |  target_id={target_id}")

        elif gesture == "Hi!":
            target_id = person["id"]   # única persona detectada
            state = State.HI
            color_state(pub=pub_rgb, color=colors["purple"])
            isInHome = False
            gc.reset_window()
            print(f"IDLE → HI  |  target_id={target_id}")

    # ── ESTADO FOLLOWING: IBVS activo + vigilar gesto Stop ───────────────────
    elif state == State.FOLLOWING:
        target = person if person and person["id"] == target_id else None

        # --- Evitar que el objetivo se pierda por el movimiento brusco de la camara ---
        if target is None:
            frame_count += 1
            if frame_count == FRAME_TOL:
                rospy.logwarn("Objetivo perdido — FOLLOWING → IDLE")
                state, target_id = State.IDLE, None
                frame_count = 0
            return
        
        frame_count = 0
        vx, vyaw = compute_ibvs(target["bbox"])
        publish_vel(pub_vel, vx, vyaw)

        if gesture == "Stop":
            state = State.STOPPED
            stop(pub=pub_vel, client=go_home)
            reset_slew_rates()
            color_state(pub=pub_rgb, color=colors["red"])
            gc.reset_window()
            print("FOLLOWING → STOPPED")

    # ── ESTADO STOPPED: quieto + vigilar Continue o Hi! ──────────────────────
    elif state == State.STOPPED:
        if person is None or person["id"] != target_id:
            rospy.logwarn("Objetivo perdido — STOPPED → IDLE")
            state, target_id = State.IDLE, None
            return

        if gesture is None:
            return

        if gesture == "Continue":
            state = State.FOLLOWING
            color_state(pub=pub_rgb, color=colors["green"])
            gc.reset_window()
            print("STOPPED → FOLLOWING")

        elif gesture == "Hi!":
            state = State.HI
            color_state(pub=pub_rgb, color=colors["purple"])
            gc.reset_window()
            print("STOPPED → HI")


if __name__ == "__main__":

    # --- Variables globales a actualizar en runtime ---    
    state = State.IDLE          # Estado actual del robot
    target_id = None            # ID de la persona objetivo
    hi_start = None             # Timestamp de entrada al estado HI
    isInHome = False            # Bandera de posición HOME
    action_executed = False     # Bandera de Acción ejecutada
    vyaw = 0.0                  # Velocidad angular
    frame_count = 0             # Conteo de frames para la tolerancia de objetivo perdido

    # --- Cargar parámetros intrínsecos de la cámara ---
    params = load_params("models/camera_params.npz")
    K_matrix = params["K"]
    dist_coeffs = params["dist"]
    img_size = params["img_size"]
    fy = K_matrix[1][1]

    # --- Calcular valores objetivo para procesar errores de velocidad (v, ω) ---
    TARGET_BBOX_H = int(np.clip(fy * PERSON_H / TARGET_DIST, None, 0.95*img_size[1]))   # Altura del bbox deseada (clipping: 95% del alto)
    cx_img = img_size[0] / 2                                                            # Centroide del frame

    map1, map2, roi_slice = build_undistort_maps(K_matrix, dist_coeffs, img_size)

    # --- YOLOv8n + BoTSORT ---
    print("\nCargando YOLOv8 ...")
    yolo = YOLO(PERSON_MODEL_PATH, task="detect")

    # --- Red de gestos Keras ---
    print("\nCargando clasificador de gestos ...")
    gesture_model = build_gesture_model(weights_path=CLASSIFIER_WEIGHTS)

    # --- Gestos a clasificar ---
    with open(CLASSES_PATH) as f:
        gesture_classes = json.load(f)

    # --- Detector de landmarks MediaPipe (modo VIDEO para timestamps continuos) ---
    print("\nCargando MediaPipe HandLandmarker ...")
    base_opts = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    mp_opts = vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    hand_detector = vision.HandLandmarker.create_from_options(mp_opts)

    bridge = CvBridge()

    # --- Inicializar clases de clasificación de gestos con ventana deslizante y slew rates limiters ---
    gc = ClassifyGesture(gesture_classes, min_conf=MIN_CONF)
    sr_lin = SlewRateLimiter(SLEW_RATE_LIN)
    sr_ang = SlewRateLimiter(SLEW_RATE_ANG)

    # --- Inicializar nodo ---
    rospy.init_node("puppypi_ibvs_controller", anonymous=False)

    # --- Subscribers y Publishers ---
    rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, callback_cam, queue_size=1)
    rospy.Subscriber("/ml_project/shutdown_signal", EmptyMsg, callback_shutdown, queue_size=1)

    pub_vel = rospy.Publisher("/puppy_control/velocity/autogait", Velocity, queue_size=1)
    pub_rgb = rospy.Publisher("/sensor/rgb_led", Led , queue_size=1)
    pub_gesture = rospy.Publisher("/ml_project/hand_gesture", String, queue_size=1)
    pub_hand_coords = rospy.Publisher("/ml_project/hand_coords", Float32MultiArray, queue_size=1)
    pub_person_track = rospy.Publisher("/ml_project/person_track", String, queue_size=1)

    # --- Service Clients ---
    rospy.wait_for_service("/puppy_control/go_home", timeout=10.0)
    go_home = rospy.ServiceProxy("/puppy_control/go_home", Empty)
    rospy.wait_for_service("/puppy_control/runActionGroup", timeout=10.0)
    action_group = rospy.ServiceProxy("/puppy_control/runActionGroup", SetRunActionName)

    print("\nControlador IBVS listo — esperando frames de /usb_cam/image_raw ...\n")
    start_time = time.time()
    rospy.spin()