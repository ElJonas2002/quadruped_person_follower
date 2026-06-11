import cv2
import rospy
from sensor_msgs.msg import Image
from project_module import load_params, undistort_image, extract_landmarks
from cv_bridge import CvBridge
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -- Convertir imagen de PuppyPi a CV2 RGB --
def callback_cam(img):
    global frame
    frame = CV_BRIDGE.imgmsg_to_cv2(img, "bgr8")

if __name__=="__main__":
    frame = None                                 # Frame actual
    CV_BRIDGE = CvBridge()                       # Puente ROS-OpenCV
    MODEL_PATH = 'models/hand_landmarker.task'   # Ruta relativa del modelo Hand Landmarks

    # -- Parámetros intrínsecos de la cámara --
    params = load_params("models/camera_params.npz")
    K_matrix = params["K"]
    dist_coeffs = params["dist"]

    # -- Opciones del modelo de landmarks --
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options = base_options,
        running_mode = vision.RunningMode.VIDEO,
        num_hands = 1
    )

    # -- Crear objeto de detección de landmarks con las opciones preestablecidas --
    detector = vision.HandLandmarker.create_from_options(options)

    # -- Inicializar nodo de ROS para obtener la imagen de la cámara --
    rospy.init_node("landmark_extractor", anonymous=False)
    rospy.Subscriber("/usb_cam/image_raw", Image, callback_cam)

    # -- Iniciar timestamp --
    start_time = time.time()

    while not rospy.is_shutdown():
        if frame is None:
            continue
        
        # -- Corregir frame distorsionado --
        undist_frame = undistort_image(frame, K_matrix, dist_coeffs)

        # -- Extraer landmarks de la mano detectada --
        lm_frame, _, _ = extract_landmarks(detector, undist_frame, start_time=start_time, draw_points=True)
        
        # -- Mostrar detección --
        cv2.imshow("Hands", lm_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()