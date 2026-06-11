import os
import json
import shutil
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# ====== CONFIGURA ESTO ======
COCO_PATH    = "datasets"
OUTPUT_PATH  = "coco_person_cleanned"
SPLITS       = ["train2017", "val2017"]

# --- Umbrales de limpieza ---
MIN_BBOX_AREA   = 300    # px² mínimos por bounding box
MIN_BBOX_SIDE   = 10     # px mínimos en ancho y alto
MAX_PERSONS     = 12     # máximo de personas por imagen
# ============================


# ─────────────────────────────────────────────
#  ETAPA 1 — Verificar integridad del archivo
# ─────────────────────────────────────────────
def is_valid_image(path: str) -> bool:
    """Devuelve True si la imagen existe y no está corrupta."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
#  ETAPA 2 — Validar bounding box
# ─────────────────────────────────────────────
def is_valid_bbox(bbox, img_w: int, img_h: int) -> bool:
    """
    Descarta bboxes que:
      - tengan coordenadas fuera de la imagen
      - sean degenerados (w o h <= 0)
      - sean demasiado pequeños (área < MIN_BBOX_AREA o lado < MIN_BBOX_SIDE)
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
        return False
    if w * h < MIN_BBOX_AREA:
        return False
    if w < MIN_BBOX_SIDE or h < MIN_BBOX_SIDE:
        return False
    return True


# ─────────────────────────────────────────────
#  Conversión bbox COCO → YOLO normalizado
# ─────────────────────────────────────────────
def coco_to_yolo(bbox, img_w: int, img_h: int):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm   = w / img_w
    h_norm   = h / img_h
    return x_center, y_center, w_norm, h_norm


# ─────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def convert_coco_to_yolo():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for split in SPLITS:
        print(f"\n{'='*50}")
        print(f"  Procesando {split}...")
        print(f"{'='*50}")

        image_dir = os.path.join(COCO_PATH, split)
        ann_file  = os.path.join(COCO_PATH, "annotations", f"instances_{split}.json")

        output_images = os.path.join(OUTPUT_PATH, "images", split)
        output_labels = os.path.join(OUTPUT_PATH, "labels", split)
        os.makedirs(output_images, exist_ok=True)
        os.makedirs(output_labels, exist_ok=True)

        with open(ann_file) as f:
            coco = json.load(f)

        # ── Obtener ID de "person" ──────────────────
        person_id = next(
            (cat["id"] for cat in coco["categories"] if cat["name"] == "person"),
            None
        )
        print(f"  ID de 'person': {person_id}")

        # ── Indexar metadatos de imágenes ───────────
        images = {img["id"]: img for img in coco["images"]}

        # ── ETAPA 2: Agrupar anotaciones válidas por imagen ──
        #    Se descartan: otras clases, instancias crowd,
        #    y bboxes que no pasen la validación geométrica.
        annotations = defaultdict(list)

        skipped_crowd    = 0
        skipped_bbox     = 0

        for ann in coco["annotations"]:
            if ann["category_id"] != person_id:
                continue

            # Descartar instancias crowd (dificultan el aprendizaje)
            if ann.get("iscrowd", 0) == 1:
                skipped_crowd += 1
                continue

            img_info = images.get(ann["image_id"])
            if img_info is None:
                continue

            # Descartar bboxes inválidos o demasiado pequeños
            if not is_valid_bbox(ann["bbox"], img_info["width"], img_info["height"]):
                skipped_bbox += 1
                continue

            annotations[ann["image_id"]].append(ann)

        print(f"  [Limpieza] Instancias crowd descartadas : {skipped_crowd}")
        print(f"  [Limpieza] Bboxes inválidos descartados : {skipped_bbox}")

        # ── ETAPA 3: Filtrar imágenes por densidad ──────────
        #    Descarta escenas con demasiadas personas (estadios,
        #    manifestaciones) que no representan el contexto de uso.
        before_density = len(annotations)
        annotations = {
            img_id: anns
            for img_id, anns in annotations.items()
            if len(anns) <= MAX_PERSONS
        }
        skipped_density = before_density - len(annotations)
        print(f"  [Limpieza] Imágenes descartadas por densidad (>{MAX_PERSONS}): {skipped_density}")

        # ── ETAPA 1 + Copia + Conversión ────────────────────
        copied          = 0
        missing         = 0
        skipped_corrupt = 0

        for img_id, anns in tqdm(annotations.items(), desc=f"  Convirtiendo {split}"):
            img_info  = images[img_id]
            file_name = img_info["file_name"]
            width     = img_info["width"]
            height    = img_info["height"]

            src_img_path = os.path.join(image_dir, file_name)
            dst_img_path = os.path.join(output_images, file_name)

            # Verificar existencia
            if not os.path.exists(src_img_path):
                missing += 1
                continue

            # ETAPA 1 — Verificar que la imagen no está corrupta
            if not is_valid_image(src_img_path):
                skipped_corrupt += 1
                continue

            # Copiar imagen al destino
            if not os.path.exists(dst_img_path):
                shutil.copy2(src_img_path, dst_img_path)
            copied += 1

            # Escribir etiquetas en formato YOLO
            label_path = os.path.join(
                output_labels,
                os.path.splitext(file_name)[0] + ".txt"   # soporta .jpg y .jpeg
            )
            with open(label_path, "w") as f:
                for ann in anns:
                    xc, yc, wn, hn = coco_to_yolo(ann["bbox"], width, height)
                    f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

        # ── Resumen del split ────────────────────────────────
        print(f"\n  ✅ {split} completado")
        print(f"     Imágenes copiadas       : {copied}")
        print(f"     Sin archivo fuente      : {missing}")
        print(f"     Corruptas descartadas   : {skipped_corrupt}")
        print(f"     Total instancias crowd  : {skipped_crowd}")
        print(f"     Total bboxes inválidos  : {skipped_bbox}")
        print(f"     Imágenes alta densidad  : {skipped_density}")
        total_removed = missing + skipped_corrupt + skipped_density
        print(f"     ─────────────────────────────")
        print(f"     Total imágenes removidas: {total_removed}")


if __name__ == "__main__":
    convert_coco_to_yolo()