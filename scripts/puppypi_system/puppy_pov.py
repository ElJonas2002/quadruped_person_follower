import rospy
import json
import time
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Float32MultiArray, Empty as EmptyMsg
from puppy_frame_processing import load_params, build_undistort_maps, undistort_image, draw_landmarks

# Obtener imagen de la camara
def callback_pov(img):
    global frame, t

    frame = bridge.compressed_imgmsg_to_cv2(img, "bgr8")
    t = time.time()

# Obtener gesto, confianza y estado
def callback_gesture(msg):
    global gesture, conf, state
    data = msg.data.split(":")
    gesture = data[0]
    conf = float(data[1])
    state = data[2]

# Obtener bbox de la mano publicada por el ibvs_controller
def callback_hand_coords(msg):
    global hand_coords
    hand_coords = msg.data if len(msg.data) == 63 else None

# Obtener track de la persona detectada por YOLO
def callback_person_track(msg):
    global person_data
    data = json.loads(msg.data)
    person_data = data if data else None

if __name__ == "__main__":
    bridge = CvBridge()
    gesture = ""
    conf = 0.0
    state = "inactivo"
    hand_coords = None
    person_data = None

    frame = None
    t = 0.0
    prev_t = 0.0
    fps = 0.0

    # Parámetros de la cámara calibrados
    params = load_params("models/camera_params.npz")
    K_matrix = params["K"]
    dist_coeffs = params["dist"]
    map1, map2, roi_slice = build_undistort_maps(K_matrix, dist_coeffs, params["img_size"])

    # Colores por estado
    STATE_COLORS = {
        "inactivo"  : (255, 0, 0),
        "saludando" : (255, 0, 188),
        "siguiendo" : (0, 255, 0),
        "pausado"   : (0, 165, 255),
    }

    # Inicializar nodo y suscribirlo a los topics correspondientes
    rospy.init_node('puppypi_pov', anonymous=False)
    rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, callback_pov, queue_size=1)
    rospy.Subscriber('/ml_project/hand_gesture', String, callback_gesture)
    rospy.Subscriber('/ml_project/hand_coords', Float32MultiArray, callback_hand_coords)
    rospy.Subscriber('/ml_project/person_track', String, callback_person_track)
    pub_shutdown = rospy.Publisher('/ml_project/shutdown_signal', EmptyMsg, queue_size=1)

    while not rospy.is_shutdown():
        if frame is None:
            continue

        undist_f = undistort_image(frame, map1, map2, roi_slice)

        # Calcular FPS
        dt = t - prev_t
        if dt > 0:
            fps = 1.0 // dt
        prev_t = t

        h, w = undist_f.shape[:2]
        state_color = STATE_COLORS.get(state, (128, 128, 128))

        # Dibujar bbox de la persona detectada por YOLO
        if person_data is not None:
            x1, y1, x2, y2 = person_data["bbox"]
            is_target = person_data["is_target"]
            cv2.rectangle(undist_f, (x1, y1), (x2, y2), state_color, thickness=2)
            id_label = f"ID:{person_data['id']}" + (" [obj]" if is_target else "")
            cv2.putText(undist_f, id_label, (x1, y2 + 15), cv2.FONT_HERSHEY_COMPLEX, 0.6, state_color, 2)

        # Dibujar landmarks de la mano si están disponibles
        if hand_coords is not None:
            undist_f = draw_landmarks(undist_f, hand_coords)

            label = f"{gesture} ({conf:.2f})" if conf > 0.0 else "Unknown Gesture"
            cv2.putText(undist_f, label, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
    
        cv2.putText(undist_f, f"FPS: {fps}", (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("PuppyPi POV", undist_f)

        if cv2.waitKey(1) & 0xFF == 27:
            pub_shutdown.publish(EmptyMsg())
            rospy.sleep(0.2)
            break
        frame = None

    cv2.destroyAllWindows()
    rospy.signal_shutdown("Cerrar Aplicación")