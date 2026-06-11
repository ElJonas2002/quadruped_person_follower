import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

# Convertir mensaje del topic a elemnto de cv2
def callback_frame(img):
    global frame
    frame = bridge.imgmsg_to_cv2(img, "bgr8")


if __name__=="__main__":
    IMG_TIMEOUT = 5.0 # Tiempo entre fotos (seg)
    IMGS_PATH = "/home/pi/camera_calibration/samples"
    TOTAL_IMGS = 15 # Número de fotos a tomar

    frame = None
    img_counter = 0
    wait_time = None
    isTimerActive = False
    bridge = CvBridge()

    rospy.init_node("chessboard_pictures", anonymous=True)
    rospy.Subscriber("/usb_cam/image_raw", Image, callback_frame)

    print("     Iniciando caputura de imagenes de chessboard para calibración...")
    time.sleep(2)
    print(f"     Imágenes a tomar: {TOTAL_IMGS} imágenes")
    time.sleep(2)
    print(f"     Tienes {IMG_TIMEOUT} segundos para posar el tablero entre fotos.")
    time.sleep(2)

    while not rospy.is_shutdown():
        if frame is None:
            continue

        # Activar timer al inciar o tras tomar una foto
        if not isTimerActive:
            print("     Timer activado! Posa el tablero")
            isTimerActive = True
            wait_time = time.time()

        # Guardar foto y reiniciar timer
        if time.time() - wait_time >= IMG_TIMEOUT:
            img_name = f"{IMGS_PATH}/chessboard_{img_counter}.png"
            cv2.imwrite(img_name, frame)
            print(f"    Imagen guardada en: {IMGS_PATH}")

            wait_time = None
            isTimerActive = False
            img_counter += 1

        cv2.imshow("PuppyPi Camera", frame)

        if img_counter == TOTAL_IMGS | (cv2.waitKey(27) & 0xFF == 27):
            break
    
    cv2.destroyAllWindows()
    print(f"    {img_counter} imágenes guardadas en: {IMGS_PATH}")