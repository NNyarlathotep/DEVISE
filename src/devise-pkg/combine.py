import numpy as np
import cv2
import os
from PIL import Image

color_mappings = {
    0: {"name": "Tree", "color": (0, 100, 0)},     # Tree
    1: {"name": "Grass", "color": (144, 238, 144)}, # Grass
    2: {"name": "Plant", "color": (34, 139, 34)}   # Plant
}

def overlay_mask_on_image(image_path, mask_npy_path, output_path):

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    mask_data = np.load(mask_npy_path)

    mask_overlay = np.zeros_like(image_np, dtype=np.uint8)

    for i, properties in color_mappings.items():
        mask = mask_data[:, :, i]
        mask = (mask == 255).astype(np.uint8) * 255

        if np.count_nonzero(mask) == 0:
            continue

        color_layer = np.full_like(image_np, properties["color"], dtype=np.uint8)

        mask_overlay = np.where(mask[:, :, None] == 255, color_layer, mask_overlay)

    alpha = 0.5
    final_result = cv2.addWeighted(image_np, 1 - alpha, mask_overlay, alpha, 0)

    cv2.imwrite(output_path, cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))

def main(image_folder, mask_folder, result_folder):

    os.makedirs(result_folder, exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        mask_filename = os.path.splitext(image_file)[0].replace('_resized', '') + "_mask.npy"
        mask_path = os.path.join(mask_folder, mask_filename)
        output_path = os.path.join(result_folder, image_file.replace('_resized', '_combined'))

        if os.path.exists(mask_path):
            print(f"✅  {image_file} and {mask_filename} combined")
            overlay_mask_on_image(image_path, mask_path, output_path)

    print("FINISH")

image_folder = "resized_folder"
mask_folder = "mask_folder"
result_folder = "combined_folder"

main(image_folder, mask_folder, result_folder)
