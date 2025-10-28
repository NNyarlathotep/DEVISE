import numpy as np
import cv2
import os
from PIL import Image

# Color mapping for vegetation types
VEGETATION_COLOR_MAP = {
    0: {"name": "Tree", "color": (0, 100, 0)},     # Dark green for trees
    1: {"name": "Grass", "color": (144, 238, 144)}, # Light green for grass
    2: {"name": "Plant", "color": (34, 139, 34)}   # Medium green for plants
}

def create_vegetation_overlay(image_path, mask_npy_path, output_path, alpha=0.5):
    """
    Create a visualization by overlaying vegetation segmentation mask on the original image.
    
    Args:
        image_path: Path to the input image
        mask_npy_path: Path to the segmentation mask (.npy file)
        output_path: Path to save the overlaid result
        alpha: Transparency factor for the overlay (0.0 to 1.0)
    """
    # Load image and mask
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    mask_data = np.load(mask_npy_path)

    # Create colored overlay for vegetation classes
    mask_overlay = np.zeros_like(image_np, dtype=np.uint8)

    for class_idx, properties in VEGETATION_COLOR_MAP.items():
        mask = mask_data[:, :, class_idx]
        mask = (mask == 255).astype(np.uint8) * 255

        if np.count_nonzero(mask) == 0:
            continue

        color_layer = np.full_like(image_np, properties["color"], dtype=np.uint8)
        mask_overlay = np.where(mask[:, :, None] == 255, color_layer, mask_overlay)

    # Blend original image with overlay
    final_result = cv2.addWeighted(image_np, 1 - alpha, mask_overlay, alpha, 0)
    cv2.imwrite(output_path, cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))

def generate_visualization_batch(image_folder, mask_folder, result_folder):
    """
    Batch process images to create vegetation overlay visualizations.
    
    Args:
        image_folder: Directory containing input images
        mask_folder: Directory containing segmentation masks (.npy files)
        result_folder: Directory to save visualization results
    """
    os.makedirs(result_folder, exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        mask_filename = os.path.splitext(image_file)[0].replace('_resized', '') + "_mask.npy"
        mask_path = os.path.join(mask_folder, mask_filename)
        output_path = os.path.join(result_folder, image_file.replace('_resized', '_combined'))

        if os.path.exists(mask_path):
            print(f"✅  Processing: {image_file} with {mask_filename}")
            create_vegetation_overlay(image_path, mask_path, output_path)

    print("✨ Visualization generation complete!")

if __name__ == "__main__":
    image_folder = "resized_folder"
    mask_folder = "mask_folder"
    result_folder = "combined_folder"

    generate_visualization_batch(image_folder, mask_folder, result_folder)


    generate_visualization_batch(image_folder, mask_folder, result_folder)
