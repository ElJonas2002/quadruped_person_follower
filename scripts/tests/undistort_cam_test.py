#import rospy
#from geometry_msgs.msg import Twist
from camera_calibration import load_params, undistort_image
import matplotlib.pyplot as plt
import cv2

CAMERA_PARAMS = "camera_params.npz"
DISTORTED_IMG = "samples/chessboard_5.png"

params = load_params("camera_params.npz")

K_matrix = params["K"]
dist_coeffs = params["dist"]
dist_img = cv2.imread(DISTORTED_IMG)
undist_img = undistort_image(dist_img, K_matrix, dist_coeffs)

fig, ax = plt.subplots(1,2,figsize=(12,10))
fig.tight_layout()

plt.subplot(121)
plt.imshow(dist_img)
plt.title("Imagen distorsionada")
plt.axis("off")
plt.subplot(122)
plt.imshow(undist_img)
plt.title("Imagen corregida")
plt.axis("off")
plt.show()