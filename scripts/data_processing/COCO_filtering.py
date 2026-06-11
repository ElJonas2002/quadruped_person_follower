import os
import json
import shutil
from tqdm import tqdm

# ====== CONFIGURA ESTO ======
COCO_PATH = "dataset"
OUTPUT_PATH = "coco_person_yolo"
SPLITS = ["train2017", "val2017"]
# ============================

def convert_coco_to_yolo():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for split in SPLITS:
        print(f"\nProcesando {split}...")

        image_dir = os.path.join(COCO_PATH, split)
        ann_file = os.path.join(COCO_PATH, "annotations", f"instances_{split}.json")

        output_images = os.path.join(OUTPUT_PATH, "images", split)
        output_labels = os.path.join(OUTPUT_PATH, "labels", split)

        os.makedirs(output_images, exist_ok=True)
        os.makedirs(output_labels, exist_ok=True)

        with open(ann_file) as f:
            coco = json.load(f)

        # Obtener ID de "person"
        person_id = None
        for cat in coco["categories"]:
            if cat["name"] == "person":
                person_id = cat["id"]
                break
        print(f"ID de 'person': {person_id}")

        # Indexar imágenes
        images = {img["id"]: img for img in coco["images"]}

        # Agrupar anotaciones por imagen
        annotations = {}
        for ann in coco["annotations"]:
            if ann["category_id"] != person_id:
                continue
            img_id = ann["image_id"]
            if img_id not in annotations:
                annotations[img_id] = []
            annotations[img_id].append(ann)

        # Procesar cada imagen
        copied, missing = 0, 0
        for img_id, anns in tqdm(annotations.items()):
            img_info = images[img_id]
            file_name = img_info["file_name"]
            width = img_info["width"]
            height = img_info["height"]

            src_img_path = os.path.join(image_dir, file_name)
            dst_img_path = os.path.join(output_images, file_name)

            # Verificar que la imagen fuente existe
            if not os.path.exists(src_img_path):
                missing += 1
                continue

            # Copiar imagen
            if not os.path.exists(dst_img_path):
                shutil.copy2(src_img_path, dst_img_path)
            copied += 1

            # Escribir anotaciones YOLO
            label_path = os.path.join(output_labels, file_name.replace(".jpg", ".txt"))
            with open(label_path, "w") as f:
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        print(f"{split} listo ✅ — Copiadas: {copied} | Sin fuente: {missing}")

if __name__ == "__main__":
    convert_coco_to_yolo()