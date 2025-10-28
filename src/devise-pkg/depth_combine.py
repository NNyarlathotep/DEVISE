import os
import numpy as np
import sqlite3

def compute_weighted_gvi(mask, depth_norm):
    mask = (mask == 255).astype(np.float32)
    weight = depth_norm.astype(np.float32)
    numerator = np.sum(mask * weight)
    denominator = np.sum(weight)
    return 0.0 if denominator == 0 else numerator / denominator

def init_db(db_path):
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

def save_to_db(conn, image_id, tree, grass, plant, total):
    cursor = conn.cursor()
    cursor.execute("""
        REPLACE INTO gvi (image_id, tree_gvi, grass_gvi, plant_gvi, total_gvi)
        VALUES (?, ?, ?, ?, ?)
    """, (image_id, tree, grass, plant, total))
    conn.commit()

def main(mask_folder, depth_folder, db_path):
    conn = init_db(db_path)
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith("_mask.npy")]
    for mask_file in mask_files:
        image_id = mask_file.replace("_mask.npy", "")
        mask_path = os.path.join(mask_folder, mask_file)
        depth_path = os.path.join(depth_folder, f"{image_id}_depth.npy")
        if not os.path.exists(depth_path):
            print(f"⚠️  Skipped {image_id}: depth not found")
            continue
        mask = np.load(mask_path)
        depth = np.load(depth_path)
        if mask.shape[:2] != depth.shape:
            print(f"⚠️  Skipped {image_id}: mask and depth size mismatch")
            continue
        tree = compute_weighted_gvi(mask[:, :, 0], depth)
        grass = compute_weighted_gvi(mask[:, :, 1], depth)
        plant = compute_weighted_gvi(mask[:, :, 2], depth)
        combined_mask = ((mask[:, :, 0] == 255) | (mask[:, :, 1] == 255) | (mask[:, :, 2] == 255)).astype(np.uint8) * 255
        total = compute_weighted_gvi(combined_mask, depth)
        save_to_db(conn, image_id, float(tree), float(grass), float(plant), float(total))
        print(f"✅  {image_id} depth information combined")
    conn.close()
    print("FINISH")

mask_folder = "mask_folder"
depth_folder = "depth_mask_folder"
db_path = "gvi_with_depth.db"

main(mask_folder, depth_folder, db_path)