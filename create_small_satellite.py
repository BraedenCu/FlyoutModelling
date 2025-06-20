#!/usr/bin/env python3
"""
Create a smaller version of the satellite image for quick testing
"""

import rasterio
import numpy as np
import os

def create_small_satellite_image():
    """Create a smaller version of the satellite image for testing"""
    
    input_path = 'satellite_data/m_3211301_nw_12_030_20230625.jp2'
    output_path = 'satellite_data/naip_quick.tif'
    
    print(f"Creating smaller satellite image for quick testing...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    with rasterio.open(input_path) as src:
        # Calculate new dimensions (reduce by factor of 8 for much faster processing)
        new_width = src.width // 8
        new_height = src.height // 8
        
        print(f"Original size: {src.width} x {src.height}")
        print(f"New size: {new_width} x {new_height}")
        print(f"Reduction factor: 8x (64x fewer pixels)")
        
        # Read the data at reduced resolution
        if src.count >= 3:
            # RGB image
            data = src.read([1, 2, 3], out_shape=(3, new_height, new_width))
        else:
            # Single band
            data = src.read(1, out_shape=(new_height, new_width))
            data = np.stack([data, data, data], axis=0)
        
        # Calculate new transform
        transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )
        
        # Save as smaller GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=new_height,
            width=new_width,
            count=3,
            dtype=data.dtype,
            crs=src.crs,
            transform=transform
        ) as dst:
            dst.write(data)
        
        # Get file sizes
        original_size = os.path.getsize(input_path) / (1024*1024)  # MB
        new_size = os.path.getsize(output_path) / (1024*1024)  # MB
        
        print(f"✅ Created smaller satellite image: {output_path}")
        print(f"Original file size: {original_size:.1f} MB")
        print(f"New file size: {new_size:.1f} MB")
        print(f"Size reduction: {original_size/new_size:.1f}x smaller")
        print(f"Use this file for quick testing by renaming it to 'naip_quick.tif'")

if __name__ == "__main__":
    create_small_satellite_image() 