#!/usr/bin/env python3
"""
Topography Download Script
Downloads DEM data from OpenTopography and stores it locally for reuse.
"""

import json
import numpy as np
import requests
import os
import tempfile
import rasterio
from pyproj import Proj, Transformer
from datetime import datetime

def load_reference_lla(json_file: str) -> tuple:
    """Load reference latitude, longitude, altitude from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'reference_lla' in data:
        ref_data = data['reference_lla']
        ref_lla = (
            ref_data['latitude'],
            ref_data['longitude'], 
            ref_data['altitude']
        )
        print(f"Loaded reference LLA: {ref_lla}")
        if 'description' in ref_data:
            print(f"Reference location: {ref_data['description']}")
        return ref_lla
    else:
        print("No reference_lla found in JSON, using default (0, 0, 0)")
        return (0.0, 0.0, 0.0)

def calculate_dem_bounds(ref_origin: tuple, area_size_km: float = 10.0) -> tuple:
    """
    Calculate DEM bounds based on reference point and desired area size.
    Args:
        ref_origin: (lat, lon, alt) reference point
        area_size_km: Size of area to cover in kilometers
    Returns:
        (west, south, east, north) bounds in degrees
    """
    lat0, lon0, _ = ref_origin
    
    # Convert km to degrees (approximate)
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 km * cos(latitude)
    lat_km_per_degree = 111.0
    lon_km_per_degree = 111.0 * np.cos(np.radians(lat0))
    
    # Calculate bounds
    half_size_lat = (area_size_km / 2) / lat_km_per_degree
    half_size_lon = (area_size_km / 2) / lon_km_per_degree
    
    west = lon0 - half_size_lon
    east = lon0 + half_size_lon
    south = lat0 - half_size_lat
    north = lat0 + half_size_lat
    
    print(f"Calculated DEM bounds: {west:.6f}, {south:.6f}, {east:.6f}, {north:.6f}")
    print(f"Coverage area: {area_size_km} km x {area_size_km} km")
    
    return west, south, east, north

def download_dem_data(bounds: tuple, api_key: str = None, dem_type: str = "COP30") -> str:
    """
    Download DEM data from OpenTopography and save to local file.
    Args:
        bounds: (west, south, east, north) in degrees
        api_key: OpenTopography API key (optional)
        dem_type: DEM type (COP30, NASADEM, SRTMGL1)
    Returns:
        Path to saved DEM file
    """
    west, south, east, north = bounds
    
    # Use provided API key or demo key
    api_key = api_key or "demoapikeyot2022"
    
    # OpenTopography API URL
    base_url = "https://portal.opentopography.org/API/globaldem"
    
    params = {
        "demtype": dem_type,
        "west": west,
        "east": east,
        "south": south,
        "north": north,
        "outputFormat": "GTiff",
        "API_Key": api_key
    }
    
    print(f"Downloading DEM from OpenTopography...")
    print(f"DEM type: {dem_type}")
    print(f"Bounds: {west:.6f}, {south:.6f}, {east:.6f}, {north:.6f}")
    
    try:
        resp = requests.get(base_url, params=params, timeout=120)
        
        if resp.status_code != 200 or b"<html" in resp.content[:20]:
            print(f"OpenTopography API error: {resp.status_code}")
            print(f"Response: {resp.text[:500]}")
            raise Exception("OpenTopography API error")
        
        # Create data directory if it doesn't exist
        data_dir = "topography_data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate filename based on bounds and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dem_{dem_type}_{west:.3f}_{south:.3f}_{east:.3f}_{north:.3f}_{timestamp}.tif"
        filepath = os.path.join(data_dir, filename)
        
        # Save DEM file
        with open(filepath, 'wb') as f:
            f.write(resp.content)
        
        print(f"DEM saved to: {filepath}")
        
        # Verify the file by reading it
        with rasterio.open(filepath) as src:
            Z = src.read(1)
            transform = src.transform
            height, width = Z.shape
            
            print(f"DEM verification:")
            print(f"  Shape: {Z.shape}")
            print(f"  Transform: {transform}")
            print(f"  Elevation range: {np.nanmin(Z):.1f} to {np.nanmax(Z):.1f} meters")
            print(f"  File size: {os.path.getsize(filepath) / (1024*1024):.1f} MB")
        
        return filepath
        
    except Exception as e:
        print(f"Error downloading DEM: {e}")
        raise

def create_metadata_file(dem_filepath: str, bounds: tuple, ref_origin: tuple, dem_type: str):
    """Create a metadata file with information about the downloaded DEM."""
    metadata = {
        "dem_file": dem_filepath,
        "dem_type": dem_type,
        "bounds": {
            "west": bounds[0],
            "south": bounds[1],
            "east": bounds[2],
            "north": bounds[3]
        },
        "reference_origin": {
            "latitude": ref_origin[0],
            "longitude": ref_origin[1],
            "altitude": ref_origin[2]
        },
        "download_date": datetime.now().isoformat(),
        "description": "DEM data downloaded from OpenTopography"
    }
    
    metadata_file = "topography_data/dem_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_file}")

def main():
    """Main function to use pre-generated topography data."""
    print("Topography Data Manager")
    print("=" * 50)
    
    # Check if we have pre-generated topography data
    data_dir = "topography_data"
    metadata_file = os.path.join(data_dir, "topography_metadata.json")
    dem_file = os.path.join(data_dir, "topography.tif")
    
    if os.path.exists(metadata_file) and os.path.exists(dem_file):
        print("Found pre-generated topography data!")
        print(f"Metadata file: {metadata_file}")
        print(f"DEM file: {dem_file}")
        
        # Load and display metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"DEM type: {metadata.get('dem_type', 'Unknown')}")
        print(f"Reference origin: {metadata.get('reference_origin', 'Unknown')}")
        print(f"Bounds: {metadata.get('bounds', 'Unknown')}")
        print(f"Download date: {metadata.get('download_date', 'Unknown')}")
        
        # Verify the DEM file
        try:
            with rasterio.open(dem_file) as src:
                Z = src.read(1)
                transform = src.transform
                height, width = Z.shape
                
                print(f"\nDEM verification:")
                print(f"  Shape: {Z.shape}")
                print(f"  Transform: {transform}")
                print(f"  Elevation range: {np.nanmin(Z):.1f} to {np.nanmax(Z):.1f} meters")
                print(f"  File size: {os.path.getsize(dem_file) / (1024*1024):.1f} MB")
        except Exception as e:
            print(f"Error reading DEM file: {e}")
            return None
        
        print("\nPre-generated topography data is ready for use!")
        print("The visualization script will use this local DEM data.")
        return dem_file
        
    else:
        print("Pre-generated topography data not found!")
        print(f"Expected files:")
        print(f"  - {metadata_file}")
        print(f"  - {dem_file}")
        print("\nPlease ensure the topography_data directory contains:")
        print("  - topography_metadata.json")
        print("  - topography.tif")
        return None

if __name__ == "__main__":
    main() 