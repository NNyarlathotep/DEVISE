import os
import cv2
import numpy as np
from PIL import Image
import torch
import sqlite3
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

# Initialize device and segmentation model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic").to(device)
model.eval()

# Vegetation class mappings with HSV filter ranges
VEGETATION_CLASS_MAP = {
    4: {"name": "Tree", "hsv": ([20, 40, 30], [95, 255, 255])},
    9: {"name": "Grass", "hsv": ([20, 40, 30], [95, 255, 255])},
    17: {"name": "Plant", "hsv": ([20, 35, 30], [92, 255, 255])}
}

def initialize_gvi_database(db_path):
    """
    Initialize SQLite database for storing GVI results.
    
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

def save_gvi_to_database(conn, image_id, tree_gvi, grass_gvi, plant_gvi, total_gvi):
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

def preprocess_image(image_path, output_folder, target_size=(4000, 2000)):
    """
    Remove black borders and resize image to target size.
    
    Args:
        image_path: Path to input image
        output_folder: Directory to save processed image
        target_size: Target dimensions (width, height)
        
    Returns:
        Path to the processed image
    """
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

def apply_vegetation_color_filter(image, mask, hsv_range):
    """
    Apply HSV color filtering to refine vegetation mask.
    
    Args:
        image: RGB image array
        mask: Binary segmentation mask
        hsv_range: Tuple of (lower_hsv, upper_hsv) bounds
        
    Returns:
        Filtered vegetation mask
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_hsv, upper_hsv = np.array(hsv_range[0]), np.array(hsv_range[1])
    green_mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    final_mask = cv2.bitwise_and(mask, mask, mask=green_mask)
    return final_mask

def compute_vegetation_indices(vegetation_mask):
    """
    Calculate Green Vegetation Index for each vegetation type.
    
    Args:
        vegetation_mask: 3-channel mask (trees, grass, plants)
        
    Returns:
        Tuple of (tree_gvi, grass_gvi, plant_gvi, total_gvi)
    """
    total_pixels = vegetation_mask.shape[0] * vegetation_mask.shape[1]
    tree_gvi = np.count_nonzero(vegetation_mask[:, :, 0]) / total_pixels
    grass_gvi = np.count_nonzero(vegetation_mask[:, :, 1]) / total_pixels
    plant_gvi = np.count_nonzero(vegetation_mask[:, :, 2]) / total_pixels
    total_gvi = tree_gvi + grass_gvi + plant_gvi
    return tree_gvi, grass_gvi, plant_gvi, total_gvi

def segment_vegetation_and_calculate_gvi(image_path, mask_output_folder, db_conn):
    """
    Perform semantic segmentation and calculate GVI for an image.
    
    Args:
        image_path: Path to preprocessed image
        mask_output_folder: Directory to save segmentation mask
        db_conn: Database connection for saving results
    """
    image_id = os.path.splitext(os.path.basename(image_path))[0].replace('_resized', '')
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    height, width = image_np.shape[:2]
    
    # Run semantic segmentation
    inputs = processor(images=image_np, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_semantic_segmentation(outputs, target_sizes=[(height, width)])[0]
    segmentation = results.cpu().numpy()
    
    # Create vegetation mask for each class
    final_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for i, (label, properties) in enumerate(VEGETATION_CLASS_MAP.items()):
        mask = (segmentation == label).astype(np.uint8) * 255
        if np.count_nonzero(mask) == 0:
            continue
        filtered_mask = apply_vegetation_color_filter(image_np, mask, properties["hsv"])
        final_mask[:, :, i] = filtered_mask
    
    # Save mask
    os.makedirs(mask_output_folder, exist_ok=True)
    output_filename = f"{image_id}_mask.npy"
    np.save(os.path.join(mask_output_folder, output_filename), final_mask)
    
    # Calculate and save GVI
    tree_gvi, grass_gvi, plant_gvi, total_gvi = compute_vegetation_indices(final_mask)
    save_gvi_to_database(db_conn, image_id, tree_gvi, grass_gvi, plant_gvi, total_gvi)

def run_vegetation_analysis_pipeline(raw_folder, resized_folder, mask_folder, gvi_db_path):
    """
    Complete pipeline for vegetation segmentation and GVI calculation.
    
    Args:
        raw_folder: Directory containing raw input images
        resized_folder: Directory to save preprocessed images
        mask_folder: Directory to save segmentation masks
        gvi_db_path: Path to SQLite database for GVI results
    """
    os.makedirs(resized_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    conn = initialize_gvi_database(gvi_db_path)

    image_files = [f for f in os.listdir(raw_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in image_files:
        raw_path = os.path.join(raw_folder, image_file)
        resized_path = preprocess_image(raw_path, resized_folder)
        segment_vegetation_and_calculate_gvi(resized_path, mask_folder, conn)
        img_id = os.path.splitext(image_file)[0]
        print(f"✅  Processed {img_id}: segmentation and GVI calculated")

    conn.close()
    print("✨ Vegetation analysis pipeline complete!")

if __name__ == "__main__":
    raw_folder = "raw_folder"
    resized_folder = "resized_folder"
    mask_folder = "mask_folder"
    gvi_db_path = "gvi.db"

    run_vegetation_analysis_pipeline(raw_folder, resized_folder, mask_folder, gvi_db_path)
