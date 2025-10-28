import os
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
depth_model.eval()

def main(input_folder, depth_mask_folder, depth_image_folder):
    os.makedirs(depth_mask_folder, exist_ok=True)
    os.makedirs(depth_image_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.endswith("_resized.jpg")]

    for image_file in image_files:
        image_id = os.path.splitext(image_file)[0].replace("_resized", "")
        image_path = os.path.join(input_folder, image_file)

        npy_output_path = os.path.join(depth_mask_folder, f"{image_id}_depth.npy")
        image_output_path = os.path.join(depth_image_folder, f"{image_id}_depth.jpg")

        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)

        inputs = depth_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = depth_model(**inputs)
        depth = outputs.predicted_depth.squeeze().cpu().numpy()
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())

        depth_resized = cv2.resize(depth_norm, original_size, interpolation=cv2.INTER_CUBIC)

        np.save(npy_output_path, depth_resized)

        depth_visual = (depth_resized * 255).astype(np.uint8)
        cv2.imwrite(image_output_path, depth_visual)

        print(f"✅  {image_id}_depth.jpg and {image_id}_depth.npy saved")

    print("FINISH")

resized_folder = "resized_folder"
depth_mask_folder = "depth_mask_folder"
depth_image_folder = "depth_image_folder"

main(resized_folder, depth_mask_folder, depth_image_folder)
