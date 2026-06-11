import rospy
import cv2
import json
import numpy as np
import time
from cv_bridge import CvBridge
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from camera_calibration import load_params, build_undistort_maps, undistort_image
from puppy_command import colors, color_state
from puppy_frame_processing import build_gesture_model, extract_landmarks, draw_landmarks, ClassifyGesture
from sensor_msgs.msg import CompressedImage
from sensor.msg import Led

# --- Relative paths ---
HAND_MODEL_PATH = "models/hand_landmarker.task"
CLASSIFIER_WEIGHTS = "models/gesture_classifier/gesture_weights_v4.npz"
CLASSES_PATH = "models/gesture_classifier/gesture_classes.json"

# ---PuppyPi Camera Params ---
CAMERA_PARAMS = load_params("models/camera_params.npz")
K_MATRIX = CAMERA_PARAMS["K"]
DIST_COEFFS = CAMERA_PARAMS["dist"]
IMG_SIZE = CAMERA_PARAMS["img_size"]

# --- Other parameters ---
MIN_CONF = 0.80
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9

state = {
    "IDLE":"blue",
    "STOPPED":"red",
    "HI":"purple",
    "FOLLOWING":"green"
}

# --- Convert the (21, 3) landmarks matrix to a 63-size array ---
def to_landmarks(flat):
    return np.asarray(flat, dtype=np.float32).reshape(-1, 21, 3)
 

# --- Convert a 63-size landmarks array to a (21,3) matrix ---
def to_flat(landmarks):
    return landmarks.reshape(-1, 21 * 3)


# --- Normalize landmarks ---
def normalize(landmarks):
    flat_input = landmarks.ndim == 2 and landmarks.shape[-1] == 63
    lm = to_landmarks(landmarks) if flat_input else landmarks.astype(np.float32).copy()
 
    # --- Center w.r.t wrist ---
    wrist = lm[:, WRIST_IDX:WRIST_IDX + 1, :]      # (N, 1, 3)
    lm = lm - wrist
 
    # --- Scaling by wrist - mid finger MCP distance ---
    ref = lm[:, MIDDLE_MCP_IDX, :]                  # (N, 3)
    scale = np.linalg.norm(ref, axis=1, keepdims=True)  # (N, 1)
    scale = np.where(scale > 1e-6, scale, 1.0)
    lm = lm / scale[:, :, None]

    return to_flat(lm) if flat_input else lm


# --- Convert frame to a 8-bits BGR CV2 frame and undistort it ---
def callback_cam(img):
    global frame
    frame = bridge.compressed_imgmsg_to_cv2(img, "bgr8")

    
if __name__=="__main__":
    frame = None
    curr_state = state["IDLE"]

    # --- Load gestures ---
    with open(CLASSES_PATH) as f:
        gesture_classes = json.load(f)

    # --- Instantiate CV Bride ---
    bridge = CvBridge()    

    # --- Build undistort maps to reduce computational cost during frame processing ---    
    map1, map2, roi_slice = build_undistort_maps(K_MATRIX, DIST_COEFFS, IMG_SIZE)

    # --- Build Keras gesture classifier model ---
    gesture_model = build_gesture_model(CLASSIFIER_WEIGHTS)

    # --- MediaPipe Hands detector (for continuous timestamps, use VIDEO mode) ---
    print("\nLoading MediaPipe Hands model ...")
    base_opts = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    mp_opts = vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    hand_detector = vision.HandLandmarker.create_from_options(mp_opts)

    # --- Initialize gestures classification class ---
    gc = ClassifyGesture(gesture_classes, min_conf=MIN_CONF)

    # --- Initialize node ---
    rospy.init_node("pid_project", anonymous=False)

    rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, callback_cam, queue_size=1)
    pub_rgb = rospy.Publisher("/sensor/rgb_led", Led , queue_size=1)

    start_time = time.time()
    
    while not rospy.is_shutdown():
        if frame is None:
            continue

        undist_frame = undistort_image(frame, map1, map2, roi_slice)
        # --- Extrar hand landmarks ---
        coords = extract_landmarks(hand_detector, undist_frame, start_time=start_time)

        # --- Classify gesture ---
        if coords is not None:
            gesture, conf = gc.update_window(gesture_model, coords)

            # --- States Machine ---
            if gesture == "Hi!" and (curr_state == state["IDLE"] or curr_state == state["STOPPED"]):
                curr_state = state["HI"]
            elif (gesture == "Follow_Me" and curr_state == state["IDLE"]) or (gesture == "Continue" and curr_state == state["STOPPED"]):
                curr_state = state["FOLLOWING"]
            elif gesture == "Stop" and curr_state == state["FOLLOWING"]:
                curr_state = state["STOPPED"]


            # --- Draw hand landmarks on frame ---
            undist_frame = draw_landmarks(undist_frame, coords)

            label = f"{gesture} ({conf:.2f})" if conf > 0.0 else "Unknown Gesture"
            cv2.putText(undist_frame, label, (10, 480 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[curr_state], 2)

        color_state(pub_rgb, colors[curr_state])
        cv2.imshow("PuppyPi POV", undist_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            color_state(pub_rgb, colors["off"])
            break

    cv2.destroyAllWindows()
    rospy.signal_shutdown("Close Program")