"""
DEVISE - Depth-Aware Vegetation Indexing System

A package for vegetation segmentation and indexing with depth awareness.
"""

__version__ = "1.0.2"
__author__ = "Cantay Caliskan, Zhizhuang Chen, Junjie Zhao, Linglan Yang, Mingzhen Zhang"
__email__ = "ccaliska@ur.rochester.edu, zchen141@u.rochester.edu, jzhao58@u.rochester.edu, lyang49@u.rochester.edu, mzhang96@u.rochester.edu"

# Import core functions with descriptive names
from .combine import (
    create_vegetation_overlay,
    generate_visualization_batch,
)

from .depth import (
    estimate_depth_for_image,
    generate_depth_maps_batch,
)

from .depth_combine import (
    calculate_depth_weighted_gvi,
    initialize_gvi_database as init_depth_gvi_db,
    save_gvi_results as save_depth_gvi_results,
    compute_depth_weighted_gvi_batch,
)

from .pipeline import (
    initialize_gvi_database,
    save_gvi_to_database,
    preprocess_image,
    apply_vegetation_color_filter,
    compute_vegetation_indices,
    segment_vegetation_and_calculate_gvi,
    run_vegetation_analysis_pipeline,
)

# Define public API
__all__ = [
    # Main pipeline functions
    "run_vegetation_analysis_pipeline",
    "generate_depth_maps_batch",
    "generate_visualization_batch",
    "compute_depth_weighted_gvi_batch",
    "run_complete_pipeline",
    "quick_vegetation_analysis",
    
    # Core vegetation analysis functions
    "segment_vegetation_and_calculate_gvi",
    "compute_vegetation_indices",
    "create_vegetation_overlay",
    "calculate_depth_weighted_gvi",
    
    # Image processing functions
    "preprocess_image",
    "apply_vegetation_color_filter",
    "estimate_depth_for_image",
    
    # Database functions
    "initialize_gvi_database",
    "save_gvi_to_database",
    "init_depth_gvi_db",
    "save_depth_gvi_results",
    
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
]

# Convenience functions for common workflows
def run_complete_pipeline(raw_folder, resized_folder, mask_folder, depth_mask_folder, 
                          depth_image_folder, result_folder, gvi_db_path, depth_db_path):
    """
    Execute the complete DEVISE pipeline with descriptive progress updates.
    
    Pipeline stages:
    1. Preprocess and segment vegetation
    2. Estimate depth maps
    3. Calculate depth-weighted GVI
    4. Generate visualization overlays
    
    Args:
        raw_folder: Input folder with raw images
        resized_folder: Output folder for preprocessed images
        mask_folder: Output folder for vegetation segmentation masks
        depth_mask_folder: Output folder for depth data (.npy files)
        depth_image_folder: Output folder for depth visualizations
        result_folder: Output folder for final overlay visualizations
        gvi_db_path: Path to standard GVI database
        depth_db_path: Path to depth-weighted GVI database
    """
    print("Starting DEVISE complete pipeline...")
    print("=" * 60)
    
    # Stage 1: Vegetation segmentation and GVI calculation
    print("\nStage 1/4: Vegetation Segmentation & GVI Calculation")
    print("-" * 60)
    run_vegetation_analysis_pipeline(raw_folder, resized_folder, mask_folder, gvi_db_path)
    
    # Stage 2: Depth estimation
    print("\nStage 2/4: Depth Map Generation")
    print("-" * 60)
    generate_depth_maps_batch(resized_folder, depth_mask_folder, depth_image_folder)
    
    # Stage 3: Depth-weighted GVI computation
    print("\nStage 3/4: Depth-Weighted GVI Calculation")
    print("-" * 60)
    compute_depth_weighted_gvi_batch(mask_folder, depth_mask_folder, depth_db_path)
    
    # Stage 4: Visualization generation
    print("\nStage 4/4: Visualization Generation")
    print("-" * 60)
    generate_visualization_batch(resized_folder, mask_folder, result_folder)
    
    print("\n" + "=" * 60)
    print("DEVISE complete pipeline finished successfully!")
    print(f"Results saved to: {result_folder}")
    print(f"Standard GVI database: {gvi_db_path}")
    print(f"Depth-weighted GVI database: {depth_db_path}")

def quick_vegetation_analysis(raw_folder, output_folder, db_path):
    """
    Simplified workflow for vegetation segmentation without depth analysis.
    
    Args:
        raw_folder: Input folder with raw images
        output_folder: Output folder for all results
        db_path: Path to GVI database file
    """
    import os
    
    resized_folder = os.path.join(output_folder, "resized")
    mask_folder = os.path.join(output_folder, "masks")
    result_folder = os.path.join(output_folder, "visualizations")
    
    print("Starting quick vegetation analysis...")
    
    # Run segmentation and GVI calculation
    run_vegetation_analysis_pipeline(raw_folder, resized_folder, mask_folder, db_path)
    
    # Generate visualizations
    generate_visualization_batch(resized_folder, mask_folder, result_folder)
    
    print(f"Analysis complete! Results in: {output_folder}")
