"""
DEVISE - Depth-Aware Vegetation Indexing System

A package for vegetation segmentation and indexing with depth awareness.
"""

__version__ = "0.1.0"
__author__ = "Zhizhuang Chen"
__email__ = "zchen141@u.rochester.edu"

# Import main functions from each module
from .combine import overlay_mask_on_image, main as combine_main
from .depth import main as depth_main
from .depth_combine import (
    compute_weighted_gvi,
    init_db,
    save_to_db,
    main as depth_combine_main
)
from .pipeline import (
    init_gvi_db,
    save_gvi_to_db,
    remove_black_borders_and_resize,
    apply_hsv_filter,
    calculate_gvi,
    process_and_generate_mask,
    main as pipeline_main
)

# Define what gets imported with "from devise import *"
__all__ = [
    # Main pipeline functions
    "pipeline_main",
    "depth_main", 
    "combine_main",
    "depth_combine_main",
    
    # Core functions
    "overlay_mask_on_image",
    "compute_weighted_gvi",
    "calculate_gvi",
    "process_and_generate_mask",
    
    # Utility functions
    "remove_black_borders_and_resize",
    "apply_hsv_filter",
    
    # Database functions
    "init_gvi_db",
    "save_gvi_to_db",
    "init_db",
    "save_to_db",
    
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
]

# Convenience functions for common workflows
def devise_pipeline(raw_folder, resized_folder, mask_folder, depth_mask_folder, 
                     depth_image_folder, result_folder, gvi_db_path, depth_db_path):
    """
    Run the complete DEVISE pipeline: preprocessing, segmentation, depth estimation, 
    and vegetation index calculation.
    
    Args:
        raw_folder: Input folder with raw images
        resized_folder: Output folder for resized images
        mask_folder: Output folder for segmentation masks
        depth_mask_folder: Output folder for depth masks (.npy files)
        depth_image_folder: Output folder for depth visualization images
        result_folder: Output folder for final combined results
        gvi_db_path: Path to GVI database file
        depth_db_path: Path to depth-weighted GVI database file
    """
    print("🚀 Starting DEVISE pipeline...")
    
    # Step 1: Run main segmentation pipeline
    print("📊 Step 1: Running segmentation pipeline...")
    pipeline_main(raw_folder, resized_folder, mask_folder, gvi_db_path)
    
    # Step 2: Generate depth maps
    print("🔍 Step 2: Generating depth maps...")
    depth_main(resized_folder, depth_mask_folder, depth_image_folder)
    
    # Step 3: Compute depth-weighted GVI
    print("⚖️ Step 3: Computing depth-weighted GVI...")
    depth_combine_main(mask_folder, depth_mask_folder, depth_db_path)
    
    # Step 4: Generate final visualizations
    print("🎨 Step 4: Generating final visualizations...")
    combine_main(resized_folder, mask_folder, result_folder)
    
    print("✅ DEVISE pipeline completed successfully!")

def devise_pipeline_without_depth(raw_folder, output_folder, db_path):
    """
    Quick segmentation workflow without depth processing.
    
    Args:
        raw_folder: Input folder with raw images
        output_folder: Output folder for all results
        db_path: Path to GVI database file
    """
    import os
    
    resized_folder = os.path.join(output_folder, "resized")
    mask_folder = os.path.join(output_folder, "masks") 
    result_folder = os.path.join(output_folder, "results")
    
    # Run segmentation pipeline
    pipeline_main(raw_folder, resized_folder, mask_folder, db_path)
    
    # Generate visualizations
    combine_main(resized_folder, mask_folder, result_folder)
    
    print(f"✅ Quick segmentation completed! Results saved to {output_folder}")