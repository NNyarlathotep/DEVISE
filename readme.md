# DEVISE - Depth-Aware Vegetation Indexing System

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DEVISE is a comprehensive Python package for vegetation analysis that combines semantic segmentation with depth estimation to provide accurate vegetation indexing. The system leverages state-of-the-art deep learning models to identify and quantify vegetation coverage in images while incorporating depth information for enhanced analysis.

## Installation

```bash
# Clone the repository
git clone https://github.com/NNyarlathotep/DEVISE.git
cd devise-pkg

# Install the package
pip install -e .
```

## Quick Start

### Segmentation With Depth

```python
import devise

# Run the complete DEVISE pipeline
devise.devise_pipeline(
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

### Segmentation Without Depth

```python
import devise

# For faster processing without depth analysis
devise.devise_pipeline_without_depth(
    raw_folder="raw_images",
    output_folder="output",
    db_path="results.db"
)
```

### Individual Module Usage

```python
import devise

# Step-by-step processing
# 1. Semantic segmentation and GVI calculation
devise.pipeline_main("raw_images", "resized", "masks", "gvi.db")

# 2. Depth estimation
devise.depth_main("resized", "depth_masks", "depth_images")

# 3. Depth-weighted GVI calculation
devise.depth_combine_main("masks", "depth_masks", "depth_gvi.db")

# 4. Generate visualization overlays
devise.combine_main("resized", "masks", "results")
```

### Advanced Usage

```python
import devise

# Custom image processing
devise.overlay_mask_on_image(
    image_path="image.jpg",
    mask_npy_path="mask.npy", 
    output_path="result.jpg"
)

# Calculate weighted GVI with custom depth data
import numpy as np
mask = np.load("vegetation_mask.npy")
depth = np.load("depth_map.npy")
weighted_gvi = devise.compute_weighted_gvi(mask[:,:,0], depth)  # For trees

# Database operations
conn = devise.init_gvi_db("custom.db")
devise.save_gvi_to_db(conn, "image_001", 0.65, 0.23, 0.12, 1.0)
```

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
- **Segmentation masks**: `.npy` files with per-pixel class predictions
- **Depth maps**: `.npy` files with normalized depth values
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