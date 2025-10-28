import os
import cv2
import numpy as np
from PIL import Image
import torch
import sqlite3
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic").to(device)
model.eval()

color_mappings = {
    4: {"name": "Tree", "hsv": ([20, 40, 30], [95, 255, 255])},
    9: {"name": "Grass", "hsv": ([20, 40, 30], [95, 255, 255])},
    17: {"name": "Plant", "hsv": ([20, 35, 30], [92, 255, 255])}
}

def init_gvi_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gvi (
            image_id TEXT PRIMARY KEY,
            tree_gvi REAL,
            grass_gvi REAL,
            plant_gvi REAL,
            total_gvi REAL
        )
    """)
    conn.commit()
    return conn

def save_gvi_to_db(conn, image_id, tree_gvi, grass_gvi, plant_gvi, total_gvi):
    cursor = conn.cursor()
    cursor.execute("""
        REPLACE INTO gvi (image_id, tree_gvi, grass_gvi, plant_gvi, total_gvi)
        VALUES (?, ?, ?, ?, ?)
    """, (image_id, tree_gvi, grass_gvi, plant_gvi, total_gvi))
    conn.commit()

def remove_black_borders_and_resize(image_path, output_folder, target_size=(4000, 2000)):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    else:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_img = img[y:y+h, x:x+w]
        resized_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_LANCZOS4)
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0] + "_resized.jpg"
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return output_path

def apply_hsv_filter(image, mask, hsv_range):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_hsv, upper_hsv = np.array(hsv_range[0]), np.array(hsv_range[1])
    green_mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    final_mask = cv2.bitwise_and(mask, mask, mask=green_mask)
    return final_mask

def calculate_gvi(final_mask):
    total_pixels = final_mask.shape[0] * final_mask.shape[1]
    tree_gvi = np.count_nonzero(final_mask[:, :, 0]) / total_pixels
    grass_gvi = np.count_nonzero(final_mask[:, :, 1]) / total_pixels
    plant_gvi = np.count_nonzero(final_mask[:, :, 2]) / total_pixels
    total_gvi = tree_gvi + grass_gvi + plant_gvi
    return tree_gvi, grass_gvi, plant_gvi, total_gvi

def process_and_generate_mask(image_path, mask_output_folder, db_conn):
    image_id = os.path.splitext(os.path.basename(image_path))[0].replace('_resized', '')
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    height, width = image_np.shape[:2]
    inputs = processor(images=image_np, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_semantic_segmentation(outputs, target_sizes=[(height, width)])[0]
    segmentation = results.cpu().numpy()
    final_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for i, (label, properties) in enumerate(color_mappings.items()):
        mask = (segmentation == label).astype(np.uint8) * 255
        if np.count_nonzero(mask) == 0:
            continue
        filtered_mask = apply_hsv_filter(image_np, mask, properties["hsv"])
        final_mask[:, :, i] = filtered_mask
    os.makedirs(mask_output_folder, exist_ok=True)
    output_filename = f"{image_id}_mask.npy"
    np.save(os.path.join(mask_output_folder, output_filename), final_mask)
    tree_gvi, grass_gvi, plant_gvi, total_gvi = calculate_gvi(final_mask)
    save_gvi_to_db(db_conn, image_id, tree_gvi, grass_gvi, plant_gvi, total_gvi)

def main(raw_folder, resized_folder, mask_folder, gvi_db_path):
    os.makedirs(resized_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    conn = init_gvi_db(gvi_db_path)

    image_files = [f for f in os.listdir(raw_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for image_file in image_files:
        raw_path = os.path.join(raw_folder, image_file)
        resized_path = remove_black_borders_and_resize(raw_path, resized_folder)
        process_and_generate_mask(resized_path, mask_folder, conn)
        img_id = os.path.splitext(image_file)[0]
        print(f"✅  {img_id}_resized.jpg and {img_id}_mask.npy saved")

    conn.close()
    print("FINISH")

raw_folder = "raw_folder"
resized_folder = "resized_folder"
mask_folder = "mask_folder"
gvi_db_path = "gvi.db"

main(raw_folder, resized_folder, mask_folder, gvi_db_path)