import os
import numpy as np
import sqlite3

def calculate_depth_weighted_gvi(vegetation_mask, depth_map):
    """
    Calculate depth-weighted Green Vegetation Index for a vegetation mask.
    
    Args:
        vegetation_mask: Binary mask where 255 indicates vegetation
        depth_map: Normalized depth map with values in [0, 1]
        
    Returns:
        Weighted GVI value (float)
    """
    mask_binary = (vegetation_mask == 255).astype(np.float32)
    weight = depth_map.astype(np.float32)
    numerator = np.sum(mask_binary * weight)
    denominator = np.sum(weight)
    return 0.0 if denominator == 0 else numerator / denominator

def initialize_gvi_database(db_path):
    """
    Initialize SQLite database for storing depth-weighted GVI results.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        Database connection object
    """
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

def save_gvi_results(conn, image_id, tree_gvi, grass_gvi, plant_gvi, total_gvi):
    """
    Save GVI results to database.
    
    Args:
        conn: Database connection object
        image_id: Unique identifier for the image
        tree_gvi: GVI value for trees
        grass_gvi: GVI value for grass
        plant_gvi: GVI value for plants
        total_gvi: Combined GVI value
    """
    cursor = conn.cursor()
    cursor.execute("""
        REPLACE INTO gvi (image_id, tree_gvi, grass_gvi, plant_gvi, total_gvi)
        VALUES (?, ?, ?, ?, ?)
    """, (image_id, tree_gvi, grass_gvi, plant_gvi, total_gvi))
    conn.commit()

def compute_depth_weighted_gvi_batch(mask_folder, depth_folder, db_path):
    """
    Batch process masks and depth maps to compute depth-weighted GVI.
    
    Args:
        mask_folder: Directory containing segmentation masks (.npy files)
        depth_folder: Directory containing depth maps (.npy files)
        db_path: Path to SQLite database for storing results
    """
    conn = initialize_gvi_database(db_path)
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith("_mask.npy")]
    
    for mask_file in mask_files:
        image_id = mask_file.replace("_mask.npy", "")
        mask_path = os.path.join(mask_folder, mask_file)
        depth_path = os.path.join(depth_folder, f"{image_id}_depth.npy")
        
        if not os.path.exists(depth_path):
            print(f"⚠️  Skipped {image_id}: depth map not found")
            continue
            
        mask = np.load(mask_path)
        depth = np.load(depth_path)
        
        if mask.shape[:2] != depth.shape:
            print(f"⚠️  Skipped {image_id}: mask and depth dimensions mismatch")
            continue
        
        # Calculate GVI for each vegetation type
        tree_gvi = calculate_depth_weighted_gvi(mask[:, :, 0], depth)
        grass_gvi = calculate_depth_weighted_gvi(mask[:, :, 1], depth)
        plant_gvi = calculate_depth_weighted_gvi(mask[:, :, 2], depth)
        
        # Calculate combined GVI
        combined_mask = ((mask[:, :, 0] == 255) | 
                        (mask[:, :, 1] == 255) | 
                        (mask[:, :, 2] == 255)).astype(np.uint8) * 255
        total_gvi = calculate_depth_weighted_gvi(combined_mask, depth)
        
        # Save to database
        save_gvi_results(conn, image_id, 
                        float(tree_gvi), float(grass_gvi), 
                        float(plant_gvi), float(total_gvi))
        
        print(f"✅  Processed {image_id}: depth-weighted GVI calculated")
    
    conn.close()
    print("✨ Depth-weighted GVI computation complete!")

if __name__ == "__main__":
    mask_folder = "mask_folder"
    depth_folder = "depth_mask_folder"
    db_path = "gvi_with_depth.db"

    compute_depth_weighted_gvi_batch(mask_folder, depth_folder, db_path)
