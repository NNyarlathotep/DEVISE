import os
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor

# Initialize device and depth estimation model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
depth_model.eval()

def estimate_depth_for_image(image_path, depth_npy_output_path, depth_vis_output_path=None):
    """
    Estimate depth map for a single image using DPT model.
    
    Args:
        image_path: Path to input image
        depth_npy_output_path: Path to save depth data as .npy file
        depth_vis_output_path: Optional path to save depth visualization as .jpg
        
    Returns:
        Normalized depth map as numpy array
    """
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)

    # Prepare inputs for model
    inputs = depth_processor(images=image, return_tensors="pt").to(device)

    # Run depth estimation
    with torch.no_grad():
        outputs = depth_model(**inputs)
    depth = outputs.predicted_depth.squeeze().cpu().numpy()
    
    # Normalize depth values to [0, 1]
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())

    # Resize to original image size
    depth_resized = cv2.resize(depth_normalized, original_size, interpolation=cv2.INTER_CUBIC)

    # Save depth data
    np.save(depth_npy_output_path, depth_resized)

    # Optionally save visualization
    if depth_vis_output_path:
        depth_visual = (depth_resized * 255).astype(np.uint8)
        cv2.imwrite(depth_vis_output_path, depth_visual)

    return depth_resized

def generate_depth_maps_batch(input_folder, depth_mask_folder, depth_image_folder):
    """
    Batch process images to generate depth maps.
    
    Args:
        input_folder: Directory containing input images
        depth_mask_folder: Directory to save depth data (.npy files)
        depth_image_folder: Directory to save depth visualizations (.jpg files)
    """
    os.makedirs(depth_mask_folder, exist_ok=True)
    os.makedirs(depth_image_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.endswith("_resized.jpg")]

    for image_file in image_files:
        image_id = os.path.splitext(image_file)[0].replace("_resized", "")
        image_path = os.path.join(input_folder, image_file)

        npy_output_path = os.path.join(depth_mask_folder, f"{image_id}_depth.npy")
        image_output_path = os.path.join(depth_image_folder, f"{image_id}_depth.jpg")

        # Generate depth map
        estimate_depth_for_image(image_path, npy_output_path, image_output_path)
        
        print(f"✅  Generated depth maps for {image_id}")

    print("✨ Depth estimation complete!")

if __name__ == "__main__":
    resized_folder = "resized_folder"
    depth_mask_folder = "depth_mask_folder"
    depth_image_folder = "depth_image_folder"

    generate_depth_maps_batch(resized_folder, depth_mask_folder, depth_image_folder)
