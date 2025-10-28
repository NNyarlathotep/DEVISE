# DEVISE - Depth-Aware Vegetation Indexing System

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-SSRN-red.svg)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5371210#paper-references-widget)

DEVISE (Depth-Aware Vegetation Indexing System) is a comprehensive Python package for vegetation analysis that combines semantic segmentation with depth estimation to provide accurate vegetation indexing. The system leverages state-of-the-art deep learning models to identify and quantify vegetation coverage in images while incorporating depth information for enhanced analysis. Applied to 1,441,746 panoramas collected every 20 m along the road networks of 60 U.S. swing counties (2004-2024), DEVISE produces the first county-month panel of street-level greenness spanning two decades. 

## Research Paper

This package implements the methodology described in our research paper:
**"DEVISE: Depth-Aware Vegetation Indexing System"**

[**Read the full paper on SSRN →**](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5371210#paper-references-widget)

## Installation

```bash
# Clone the repository
git clone https://github.com/NNyarlathotep/DEVISE.git
cd devise-pkg

# Install the package
pip install -e .
```

## Quick Start

### Complete Pipeline with Depth Analysis

```python
import devise

# Run the complete DEVISE pipeline
devise.run_complete_pipeline(
    raw_folder="raw_images",           # Input images
    resized_folder="resized",          # Preprocessed images
    mask_folder="masks",               # Segmentation masks
    depth_mask_folder="depth_masks",   # Depth data (.npy files)
    depth_image_folder="depth_images", # Depth visualizations
    result_folder="results",           # Final combined results
    gvi_db_path="gvi.db",             # GVI database
    depth_db_path="depth_gvi.db"      # Depth-weighted GVI database
)
```

### Quick Analysis Without Depth

```python
import devise

# For faster processing without depth analysis
devise.quick_vegetation_analysis(
    raw_folder="raw_images",
    output_folder="output",
    db_path="results.db"
)
```

### Individual Module Usage

```python
import devise

# Step-by-step processing with new function names

# 1. Vegetation segmentation and GVI calculation
devise.run_vegetation_analysis_pipeline(
    "raw_images", "resized", "masks", "gvi.db"
)

# 2. Depth map generation
devise.generate_depth_maps_batch(
    "resized", "depth_masks", "depth_images"
)

# 3. Depth-weighted GVI calculation
devise.compute_depth_weighted_gvi_batch(
    "masks", "depth_masks", "depth_gvi.db"
)

# 4. Generate visualization overlays
devise.generate_visualization_batch(
    "resized", "masks", "results"
)
```

### Advanced Usage

```python
import devise
import numpy as np

# Process a single image with custom parameters
devise.create_vegetation_overlay(
    image_path="image.jpg",
    mask_npy_path="mask.npy", 
    output_path="result.jpg",
    alpha=0.6  # Adjust transparency
)

# Estimate depth for a single image
devise.estimate_depth_for_image(
    image_path="image.jpg",
    depth_npy_output_path="depth.npy",
    depth_vis_output_path="depth_vis.jpg"
)

# Calculate depth-weighted GVI for custom data
mask = np.load("vegetation_mask.npy")
depth = np.load("depth_map.npy")
tree_gvi = devise.calculate_depth_weighted_gvi(mask[:,:,0], depth)
grass_gvi = devise.calculate_depth_weighted_gvi(mask[:,:,1], depth)
plant_gvi = devise.calculate_depth_weighted_gvi(mask[:,:,2], depth)

# Database operations
conn = devise.initialize_gvi_database("custom.db")
devise.save_gvi_to_database(conn, "image_001", 0.65, 0.23, 0.12, 1.0)
conn.close()
```

## API Reference

### Main Pipeline Functions

- **`run_complete_pipeline(...)`** - Execute full DEVISE workflow with depth analysis
- **`quick_vegetation_analysis(...)`** - Fast vegetation analysis without depth
- **`run_vegetation_analysis_pipeline(...)`** - Segmentation and GVI calculation
- **`generate_depth_maps_batch(...)`** - Batch depth estimation
- **`compute_depth_weighted_gvi_batch(...)`** - Batch depth-weighted GVI
- **`generate_visualization_batch(...)`** - Batch visualization generation

### Core Analysis Functions

- **`segment_vegetation_and_calculate_gvi(...)`** - Segment single image and compute GVI
- **`compute_vegetation_indices(...)`** - Calculate GVI from mask
- **`calculate_depth_weighted_gvi(...)`** - Compute depth-weighted GVI
- **`create_vegetation_overlay(...)`** - Create visualization overlay

### Image Processing Functions

- **`preprocess_image(...)`** - Remove borders and resize image
- **`apply_vegetation_color_filter(...)`** - Apply HSV filtering
- **`estimate_depth_for_image(...)`** - Estimate depth for single image

### Database Functions

- **`initialize_gvi_database(...)`** - Create/connect to GVI database
- **`save_gvi_to_database(...)`** - Save GVI results to database
- **`init_depth_gvi_db(...)`** - Initialize depth-weighted GVI database
- **`save_depth_gvi_results(...)`** - Save depth GVI results

## Package Structure

```
devise/
├── combine.py          # Image overlay and visualization generation
├── depth.py           # Monocular depth estimation using DPT
├── depth_combine.py    # Depth-weighted GVI calculation
├── pipeline.py         # Main segmentation and GVI pipeline
├── print_gvi.py       # Database query utilities for standard GVI
├── print_gvi_with_depth.py  # Database query utilities for depth GVI
└── __init__.py         # Package entry point with convenience functions
```

## Models Used

- **Segmentation**: `facebook/mask2former-swin-large-ade-semantic`
- **Depth Estimation**: `Intel/dpt-large`

Both models are automatically downloaded from Hugging Face Hub on first use.

## Output Format

### Database Schema
The system stores results in SQLite databases with the following structure:

```sql
CREATE TABLE gvi (
    image_id TEXT PRIMARY KEY,
    tree_gvi REAL,      -- Green Vegetation Index for trees
    grass_gvi REAL,     -- Green Vegetation Index for grass  
    plant_gvi REAL,     -- Green Vegetation Index for plants
    total_gvi REAL      -- Combined vegetation index
);
```

### File Outputs
- **Segmentation masks**: `.npy` files with per-pixel class predictions (3 channels: trees, grass, plants)
- **Depth maps**: `.npy` files with normalized depth values [0, 1]
- **Visualizations**: `.jpg` overlay images showing segmentation results
- **Databases**: `.db` files with GVI measurements

## Requirements

- Python 3.12+
- PyTorch 1.9+
- Transformers 4.15+
- OpenCV 4.5+
- NumPy 1.21+
- Pillow 8.0+
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
**Cantay Caliskan** 
Email: ccaliska@ur.rochester.edu

**Zhizhuang Chen**  
Email: zchen141@u.rochester.edu  

**Junjie Zhao** 
Email: jzhao58@u.rochester.edu

**Linglan Yang** 
Email: lyang49@u.rochester.edu

**Mingzhen Zhang** 
Email: mzhang96@u.rochester.edu