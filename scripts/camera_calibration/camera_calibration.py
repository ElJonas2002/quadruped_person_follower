"""
Calibración de cámara usando el modelo pinhole.
"""

import json
from pathlib import Path
import cv2
import numpy as np

# ── CONFIGURACIÓN ──────────────────────────────────────────────────────
IMAGES_DIR   = Path("samples")  # carpeta con las fotos del tablero
BOARD_COLS   = 7                     # esquinas internas horizontales
BOARD_ROWS   = 7                     # esquinas internas verticales
SQUARE_SIZE  = 7.6                   # Tamaño de los cuadrados del tablero (mm)
OUTPUT       = Path("camera_params") # nombre base de los archivos de salida
# ───────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Calibración
# ---------------------------------------------------------------------------

def find_chessboard_points(image_paths, board_size=(7,6), square_size=None, show_corners=True):
    """
    Detecta esquinas del tablero de ajedrez en cada imagen y devuelve los
    puntos 3D del mundo y los puntos 2D del plano imagen.

    Args:
        image_paths:     Lista de rutas a las imágenes de calibración.
        board_size:      (columnas, filas) de esquinas internas del tablero.
        square_size:     Tamaño real de cada cuadrado (mm). Escala los puntos 3D.
        show_corners:    Si True, muestra las esquinas detectadas en pantalla.

    Returns:
        objpoints: Lista de arrays de puntos 3D (uno por imagen válida).
        imgpoints: Lista de arrays de puntos 2D (uno por imagen válida).
        img_size:  (ancho, alto) de las imágenes en píxeles.
    """
    cols, rows = board_size

    # Puntos 3D del patrón en su sistema de coordenadas propio (Z=0 siempre).
    # Ejemplo para board_size=(7,6): (0,0,0),(1,0,0),...,(6,5,0)
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    # Escalar a distancias reales (útil para reconstrucción 3D y distancias)
    if square_size is not None:
        objp *= square_size

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []
    imgpoints = []
    img_size = (0, 0)
    valid_count = 0

    print(f"\nBuscando tablero {cols}×{rows} en {len(image_paths)} imágenes...")

    # Leer cada imagen del directorio y encontrar las esquinas internas
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  [WARN] No se pudo leer: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = (gray.shape[1], gray.shape[0])  # (width, height)
        
        found, corners = cv2.findChessboardCornersSB(gray, board_size, None)

        if found:
            # Refina las esquinas a nivel subpíxel
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            valid_count += 1

            # Dibujar esquinas
            if show_corners:
                cv2.drawChessboardCorners(img, board_size, corners_refined, found)
                cv2.imshow(f"Corners: {path.name}", img)
                cv2.waitKey(400)

            print(f"  [OK]  {path.name}")
        else:
            print(f"  [--]  {path.name}  (tablero no detectado)")

    if show_corners:
        cv2.destroyAllWindows()

    print(f"\nImágenes válidas: {valid_count}/{len(image_paths)}")
    return objpoints, imgpoints, img_size


def calibrate(objpoints, imgpoints, img_size):
    """
    Ejecuta la calibración de la cámara y devuelve todos los parámetros.
    Returns:
      Diccionario con:
          K:           Matriz intrínseca 3x3.
          dist:        Coeficientes de distorsión [k1,k2,p1,p2,k3].
          rvecs:       Vectores de rotación por imagen (extrínsecos).
          tvecs:       Vectores de traslación por imagen (extrínsecos).
          rms_error:   Error de reproyección RMS (píxeles); <1.0 es bueno.
          img_size:    (ancho, alto) usado en la calibración.
    """
    if len(objpoints) < 4:
        raise ValueError(
            f"Se necesitan al menos 4 imágenes válidas; solo hay {len(objpoints)}."
        )

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    return {
        "K": K,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "rms_error": rms,
        "img_size": img_size,
    }


def compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, dist):
    """
    Calcula el error de reproyección por imagen.

    El error de reproyección mide cuánto difieren los puntos 2D reales de los
    puntos que obtenemos proyectando los puntos 3D con los parámetros estimados.
    Un valor <1.0 px indica una calibración de buena calidad.
    """
    per_image_errors = []
    for i, (obj, img) in enumerate(zip(objpoints, imgpoints)):
        projected, _ = cv2.projectPoints(obj, rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(img, projected, cv2.NORM_L2) / len(projected)
        per_image_errors.append(err)

    mean_error = float(np.mean(per_image_errors))
    return mean_error, per_image_errors


# ---------------------------------------------------------------------------
# Guardar y cargar parámetros
# ---------------------------------------------------------------------------

def save_params(params, output_path, square_size=None):
    """
    Guarda los parámetros de calibración en dos formatos:
      - .npz  : binario NumPy, ideal para cargar en otros scripts Python.
      - .json : legible por humanos, compatible con cualquier lenguaje.
    """
    npz_path = output_path.with_suffix(".npz")
    save_data = {
        "K": params["K"],
        "dist": params["dist"],
        "img_size": params["img_size"],
        "rms_error": params["rms_error"],
    }
    if square_size is not None:
        save_data["square_size_mm"] = square_size

    np.savez(npz_path, **save_data)
    print(f"\nParámetros guardados (NumPy): {npz_path}")

    # JSON: convierte arrays a listas para serialización
    json_path = output_path.with_suffix(".json")
    json_data = {
        "img_size": list(params["img_size"]),
        "rms_error_px": round(float(params["rms_error"]), 6),
        "intrinsic_matrix_K": params["K"].tolist(),
        "distortion_coefficients": params["dist"].flatten().tolist(),
        "description": {
            "K[0][0]": "fx — distancia focal eje X (píxeles)",
            "K[1][1]": "fy — distancia focal eje Y (píxeles)",
            "K[0][2]": "cx — punto principal eje X (píxeles)",
            "K[1][2]": "cy — punto principal eje Y (píxeles)",
            "dist":    "[k1, k2, p1, p2, k3] — coef. de distorsión",
        },
    }
    if square_size is not None:
        json_data["square_size_mm"] = square_size

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Parámetros guardados (JSON):  {json_path}")


def load_params(npz_path):
    """
    Carga los parámetros de calibración desde un archivo .npz.
    """
    data = np.load(str(npz_path))
    params = {
        "K": data["K"],
        "dist": data["dist"],
        "img_size": tuple(data["img_size"].tolist()),
        "rms_error": float(data["rms_error"]),
    }
    if "square_size_mm" in data:
        params["square_size_mm"] = float(data["square_size_mm"])

    return params

# ---------------------------------------------------------------------------
# Corrección de distorsión
# ---------------------------------------------------------------------------

def build_undistort_maps(K, dist, img_size):
    """
    Precomputa los mapas de corrección de distorsión. Llamar una sola vez al inicio.
    Usar con undistort_image() para corregir frames a alta velocidad con cv2.remap.

    Args:
        K: Matriz intrínseca calibrada
        dist: Arreglo numpy de coeficientes de distorsión
        img_size: Tamaño de la imagen (w, h)
    Returns:
        map1, map2: Mapas de remapeo para cv2.remap
        roi_slice: Slice numpy para recortar bordes negros (o None)
    """
    w, h = img_size
    K_optimal, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K_optimal, (w, h), cv2.CV_16SC2)
    x, y, rw, rh = roi
    roi_slice = (slice(y, y + rh), slice(x, x + rw)) if rw > 0 and rh > 0 else None
    return map1, map2, roi_slice


def undistort_image(img, map1, map2, roi_slice=None):
    """
    Aplica corrección de distorsión usando mapas precomputados (cv2.remap).
    Mucho más rápido que cv2.undistort por frame — el costo de getOptimalNewCameraMatrix
    se paga solo una vez en build_undistort_maps.

    Args:
        img: Frame BGR de 8 bits a corregir.
        map1, map2: Mapas obtenidos de build_undistort_maps.
        roi_slice: Slice de recorte obtenido de build_undistort_maps.
    """
    undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    return undistorted[roi_slice] if roi_slice is not None else undistorted


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def print_report(params, per_image_errors):
    K = params["K"]
    dist = params["dist"].flatten()

    print("\n" + "=" * 55)
    print("  RESULTADOS DE CALIBRACIÓN")
    print("=" * 55)
    print(f"\n  Tamaño imagen : {params['img_size'][0]} × {params['img_size'][1]} px")
    print(f"  Error RMS     : {params['rms_error']:.4f} px  ", end="")
    print("✓ bueno" if params["rms_error"] < 1.0 else "⚠ revisar imágenes")

    print("\n  Matriz intrínseca K:")
    for row in K:
        print("   ", "  ".join(f"{v:10.4f}" for v in row))

    print(f"\n  Distancia focal : fx={K[0,0]:.2f} px,  fy={K[1,1]:.2f} px")
    print(f"  Punto principal : cx={K[0,2]:.2f} px,  cy={K[1,2]:.2f} px")
    print(f"\n  Coeficientes de distorsión:")
    labels = ["k1", "k2", "p1", "p2", "k3"]
    for label, val in zip(labels, dist):
        print(f"    {label} = {val:.6f}")

    print("\n  Error de reproyección por imagen:")
    for i, err in enumerate(per_image_errors):
        bar = "█" * int(err * 20)
        print(f"    img {i+1:02d}: {err:.4f} px  {bar}")

    print("=" * 55 + "\n")


# -----------------------------------------------------------------------------------------------
# Ejecución: Hallar puntos -> Calibrar -> Error de reproyección -> Reportar y guardar resultados
# -----------------------------------------------------------------------------------------------

# if __name__ == "__main__":

#     image_paths = sorted(list(IMAGES_DIR.glob("*.png")))

#     if not image_paths:
#         raise FileNotFoundError(f"No se encontraron imágenes en '{IMAGES_DIR}'")

#     objpoints, imgpoints, img_size = find_chessboard_points(
#         image_paths,
#         board_size=(BOARD_COLS, BOARD_ROWS),
#         square_size=SQUARE_SIZE,
#     )

#     params = calibrate(objpoints, imgpoints, img_size)

#     _, per_image_errors = compute_reprojection_errors(
#         objpoints, imgpoints,
#         params["rvecs"], params["tvecs"],
#         params["K"], params["dist"],
#     )

#     print_report(params, per_image_errors)
#     save_params(params, OUTPUT, square_size=SQUARE_SIZE)