import rospy
import cv2
import mediapipe as mp
import time
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Callback al leer un mensaje del topic image_raw (camara de PuppyPi)
def callback_camara(msg):
    # Procesar mensaje (frame)
    frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp = int((time.time() - start_time) * 1000)
    result    = detector.detect_for_video(mp_image, timestamp)

    if not result.hand_landmarks:
        return
    
    xs = []
    ys = []
    coords = []

    # Extraer landmarks de la mano (x,y,z)
    for lm in result.hand_landmarks[0]:
        coords.extend([lm.x, lm.y, lm.z])
        xs.append(lm.x)
        ys.append(lm.y)

    # Extraer coordenadas normalizadas del bbox de la mano (x1, y1, x2, y2)
    bbox = [min(xs), min(ys), max(xs), max(ys)]
    
    # Crear y publicar landmarks de la mano (topic que conecta con el clasificador de Keras en PuppyPi)
    msg_landmarks = Float32MultiArray()
    msg_landmarks.data = coords
    pub_landmarks.publish(msg_landmarks)
    print(f"  Landmarks publicados ({len(coords)} features)")

    # Crear y publicar coordenadas del bbox de la mano
    msg_bbox = Float32MultiArray()
    msg_bbox.data = bbox
    pub_bbox.publish(msg_bbox)


if __name__ == "__main__":
    # Modelo de landmarks de la mano
    HAND_MODEL = "/models/hand_landmarker.task"

    # Configuración del modelo
    base_options = python.BaseOptions(model_asset_path=HAND_MODEL)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )
    # Crear objeto del detector
    detector = vision.HandLandmarker.create_from_options(options)

    bridge = CvBridge()         # Puente de mensaje de imagen de ROS a CV2
    start_time = time.time()
    pub_landmarks = None
    pub_bbox = None
    
    # Inicializar nodo
    rospy.init_node('landmark_extractor', anonymous=True)                                 # Inicializar nodo
    pub_landmarks = rospy.Publisher('/hand_landmarks', Float32MultiArray, queue_size=10)  # Declarar publisher de landmarks
    pub_bbox = rospy.Publisher('/hand_bbox', Float32MultiArray, queue_size=10)            # Declarar publisher de bbox
    rospy.Subscriber('/usb_cam/image_raw', Image, callback_camara)                        # Declarar subscriber

    print("\n✓ Extractor de landmarks corriendo...")
    print("  Escuchando /usb_cam/image_raw")
    print("  Publicando en /hand_landmarks\n")

    rospy.spin()