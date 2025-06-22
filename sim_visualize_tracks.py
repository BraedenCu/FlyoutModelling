#!/usr/bin/env python3
"""
Trajectory Visualization Suite using PyVista and OpenTopography
A complete end-to-end visualization system for displaying trajectory tracks
with high-fidelity topographic data.
"""

import json
import numpy as np
import pyvista as pv
import requests
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
import warnings
import rasterio
from rasterio.transform import from_origin
from pyproj import Proj, Transformer
import platform
import subprocess
import signal
import sys
import atexit
import argparse

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Global cleanup variables
cleanup_required = False
plotter_instance = None

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print(f"\nReceived signal {signum} - cleaning up...")
    cleanup_and_exit()

def cleanup_and_exit():
    """Clean up resources and exit gracefully."""
    global cleanup_required, plotter_instance
    
    if cleanup_required:
        print("🧹 Cleaning up resources...")
        
        # Close PyVista plotter if it exists
        if plotter_instance is not None:
            try:
                plotter_instance.close()
                print("PyVista plotter closed")
            except:
                pass
        
        # Close any matplotlib figures
        try:
            plt.close('all')
            print("Matplotlib figures closed")
        except:
            pass
        
        # Force garbage collection
        import gc
        gc.collect()
        print("Memory cleaned up")
    
    print("Exiting gracefully")
    sys.exit(0)

def register_cleanup(plotter=None):
    """Register cleanup functions for graceful exit."""
    global cleanup_required, plotter_instance
    cleanup_required = True
    plotter_instance = plotter
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup on normal exit
    atexit.register(cleanup_and_exit)

# WSLg-specific configuration for Windows 11
def setup_wslg_display():
    """Configure display settings for Windows 11 WSLg."""
    # Check if running in WSLg
    is_wslg = False
    try:
        with open('/proc/version', 'r') as f:
            content = f.read().lower()
            if 'microsoft' in content and 'wslg' in content:
                is_wslg = True
    except:
        pass
    
    if is_wslg:
        print("WSLg detected (Windows 11) - native graphics support available!")
        # WSLg provides native graphics support, no additional configuration needed
        os.environ['DISPLAY'] = ':0'
        return True
    else:
        print("Not running in WSLg - using standard display configuration")
        return True

def configure_pyvista_for_wslg():
    """Configure PyVista settings optimized for Windows 11 WSLg."""
    # Check if running in WSLg
    is_wslg = False
    try:
        with open('/proc/version', 'r') as f:
            content = f.read().lower()
            if 'microsoft' in content and 'wslg' in content:
                is_wslg = True
    except:
        pass
    
    if is_wslg:
        print("Configuring PyVista for Windows 11 WSLg...")
        
        # WSLg provides excellent graphics support - use optimal settings
        pv.global_theme.window_size = [1280, 720]  # Optimal window size for WSLg
        
        print("PyVista configured for Windows 11 WSLg with optimal settings")
    else:
        print("Not running in WSLg - using default PyVista settings")

@dataclass
class TrajectoryPoint:
    """Represents a single point in a trajectory."""
    time: float
    position: Tuple[float, float, float]  # ENU coordinates
    velocity: Tuple[float, float, float]  # ENU velocity

@dataclass
class Trajectory:
    """Represents a complete trajectory."""
    id: int
    points: List[TrajectoryPoint]

@dataclass
class ProtectedRegion:
    """Represents a cylindrical protected region."""
    id: int
    centroid: Tuple[float, float, float]  # ENU coordinates (x, y, z)
    radius: float  # radius in meters
    height_limit: float  # maximum height in meters
    name: str  # descriptive name

@dataclass
class Missile:
    """Represents a missile trajectory."""
    id: int
    points: List[TrajectoryPoint]

@dataclass
class CollisionEvent:
    """Represents a collision event between trajectories/missiles."""
    time: float
    position: Tuple[float, float, float]  # ENU coordinates of collision
    participants: List[int]  # IDs of colliding objects
    explosion_duration: float = 2.0  # Duration of explosion animation in seconds
    explosion_radius: float = 10.0  # Maximum radius of explosion effect

@dataclass
class Radar:
    """Represents a radar station."""
    id: int
    emplacement: Tuple[float, float, float]  # ENU coordinates (x, y, z)
    type: str  # Type of resource (e.g., "radar")
    description: str  # Descriptive name

class SatelliteImageryManager:
    """Manages satellite imagery overlay for topography visualization."""
    
    def __init__(self, ref_origin=(0.0, 0.0, 0.0)):
        self.ref_origin = ref_origin
        self.satellite_texture = None
        self.satellite_bounds = None
        self.transformer_enu2llh = None
        self.transformer_llh2enu = None
        self._setup_transformers()
        
    def _setup_transformers(self):
        lat0, lon0, h0 = self.ref_origin
        # WGS84 geodetic to ENU local tangent plane
        self.transformer_llh2enu = Transformer.from_crs(
            "EPSG:4326",  # WGS84 lat/lon
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +ellps=WGS84",  # Local ENU
            always_xy=True
        )
        
        # ENU local tangent plane to WGS84 geodetic
        self.transformer_enu2llh = Transformer.from_crs(
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +ellps=WGS84",  # Local ENU
            "EPSG:4326",  # WGS84 lat/lon
            always_xy=True
        )
    
    def load_satellite_imagery(self, image_path: str) -> bool:
        """
        Load satellite imagery from file and create PyVista texture.
        Args:
            image_path: Path to satellite image file (GeoTIFF, BIL, JP2, etc.)
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Loading satellite imagery from: {image_path}")
            
            # Handle different file formats
            if image_path.lower().endswith(('.jp2', '.jpeg2000')):
                # Use rasterio for JPEG2000 files
                with rasterio.open(image_path) as src:
                    if src.count >= 3:
                        # RGB imagery
                        sat_img = src.read([1, 2, 3])  # RGB bands
                        print(f"Loaded RGB satellite imagery: {sat_img.shape}")
                    else:
                        # Single band - convert to RGB (grayscale)
                        sat_img = src.read(1)
                        sat_img = np.stack([sat_img, sat_img, sat_img], axis=0)
                        print(f"Loaded single-band satellite imagery, converted to RGB: {sat_img.shape}")
                    
                    # Get georeferencing information
                    transform = src.transform
                    bounds = src.bounds
                    self.satellite_bounds = bounds
                    
                    print(f"Satellite image bounds: {bounds}")
                    print(f"Satellite image transform: {transform}")
            else:
                # Use rasterio for other formats
                with rasterio.open(image_path) as src:
                    # Check if it's RGB imagery (3 bands) or single band
                    if src.count >= 3:
                        # RGB imagery
                        sat_img = src.read([1, 2, 3])  # RGB bands
                        print(f"Loaded RGB satellite imagery: {sat_img.shape}")
                    else:
                        # Single band - convert to RGB (grayscale)
                        sat_img = src.read(1)
                        sat_img = np.stack([sat_img, sat_img, sat_img], axis=0)
                        print(f"Loaded single-band satellite imagery, converted to RGB: {sat_img.shape}")
                    
                    # Get georeferencing information
                    transform = src.transform
                    bounds = src.bounds
                    self.satellite_bounds = bounds
                    
                    print(f"Satellite image bounds: {bounds}")
                    print(f"Satellite image transform: {transform}")
            
            print(f"Original data type: {sat_img.dtype}")
            print(f"Original value ranges:")
            for i in range(sat_img.shape[0]):
                band = sat_img[i]
                print(f"  Band {i+1}: {band.min()} to {band.max()}")
            
            # Ensure we have uint8 data for PyVista
            if sat_img.dtype != np.uint8:
                if sat_img.dtype == np.uint16:
                    # Scale from uint16 to uint8
                    sat_img = (sat_img / 256).astype(np.uint8)
                else:
                    # For other types, normalize to uint8
                    sat_img = sat_img.astype(np.float32)
                    for i in range(sat_img.shape[0]):
                        band = sat_img[i]
                        band_min, band_max = np.nanmin(band), np.nanmax(band)
                        if band_max > band_min:
                            sat_img[i] = (band - band_min) / (band_max - band_min)
                    sat_img = (sat_img * 255).astype(np.uint8)
            
            # Handle NaN values
            sat_img = np.nan_to_num(sat_img, nan=0).astype(np.uint8)
            
            # Transpose for PyVista: (bands, height, width) -> (height, width, bands)
            sat_img = np.moveaxis(sat_img, 0, -1)
            
            print(f"Final image shape: {sat_img.shape}")
            print(f"Final data type: {sat_img.dtype}")
            print(f"Final value range: {sat_img.min()} to {sat_img.max()}")
            
            # Create PyVista texture
            self.satellite_texture = pv.numpy_to_texture(sat_img)
            
            print(f"Successfully created satellite texture: {sat_img.shape}")
            return True
            
        except Exception as e:
            print(f"Error loading satellite imagery: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_texture_coordinates(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Calculate texture coordinates for satellite imagery overlay.
        Args:
            X, Y: ENU coordinate arrays
        Returns:
            Texture coordinates array (u, v) in [0, 1] range
        """
        if self.satellite_bounds is None:
            # Fallback: use normalized coordinates
            u = (X - X.min()) / (X.max() - X.min())
            v = (Y - Y.min()) / (Y.max() - Y.min())
            return np.column_stack([u.flatten(), v.flatten()])
        
        try:
            # For now, use simple normalized coordinates since coordinate systems don't match
            # This will map the satellite image to cover the entire topography area
            u = (X - X.min()) / (X.max() - X.min())
            v = (Y - Y.min()) / (Y.max() - Y.min())
            
            # Clamp to [0, 1] range
            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)
            
            return np.column_stack([u.flatten(), v.flatten()])
            
        except Exception as e:
            print(f"Error calculating texture coordinates: {e}")
            # Fallback: use normalized coordinates
            u = (X - X.min()) / (X.max() - X.min())
            v = (Y - Y.min()) / (Y.max() - Y.min())
            return np.column_stack([u.flatten(), v.flatten()])

class TopographyManager:
    """Manages downloading and processing topographic data from OpenTopography."""
    
    def __init__(self, api_key=None, dem_type="COP30", ref_origin=(0.0, 0.0, 0.0)):
        self.api_key = api_key or os.environ.get("OPENTOPO_API_KEY", None)
        self.base_url = "https://portal.opentopography.org/API/globaldem"
        self.dem_type = dem_type  # e.g., "COP30", "NASADEM", "SRTMGL1"
        self.last_dem_path = None
        self.ref_origin = ref_origin  # (lat0, lon0, h0)
        self.transformer_enu2llh = None
        self.transformer_llh2enu = None
        self.satellite_manager = SatelliteImageryManager(ref_origin=ref_origin)
        self._setup_transformers()
        
    def _setup_transformers(self):
        lat0, lon0, h0 = self.ref_origin
        # Create ENU projection centered at reference point
        self.enu_proj = Proj(proj='aeqd', lat_0=lat0, lon_0=lon0, ellps='WGS84')
        
        # WGS84 geodetic to ENU local tangent plane
        self.transformer_llh2enu = Transformer.from_crs(
            "EPSG:4326",  # WGS84 lat/lon
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +ellps=WGS84",  # Local ENU
            always_xy=True
        )
        
        # ENU local tangent plane to WGS84 geodetic
        self.transformer_enu2llh = Transformer.from_crs(
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +ellps=WGS84",  # Local ENU
            "EPSG:4326",  # WGS84 lat/lon
            always_xy=True
        )
    
    def enu_to_latlon(self, x, y, z):
        """Convert ENU coordinates to latitude/longitude."""
        if self.transformer_enu2llh is None:
            self._setup_transformers()
        
        # Use the transformer to convert ENU to lat/lon
        lon, lat = self.transformer_enu2llh.transform(x, y)
        return lat, lon
    
    def latlon_to_enu(self, lat, lon, alt=0.0):
        """Convert latitude/longitude to ENU coordinates."""
        if self.transformer_llh2enu is None:
            self._setup_transformers()
        
        # Use the transformer to convert lat/lon to ENU
        x, y = self.transformer_llh2enu.transform(lon, lat)
        return x, y, alt
    
    def load_local_topography_data(self) -> Optional[tuple]:
        """
        Load pre-generated topography data from local files.
        Returns:
            (X, Y, Z) meshgrid in ENU meters, or None if failed
        """
        try:
            # Load metadata to get the DEM file path and reference origin
            metadata_file = "topography_data/topography_metadata.json"
            if not os.path.exists(metadata_file):
                print(f"Metadata file not found: {metadata_file}")
                return None
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Get the DEM file path
            dem_file = metadata.get("dem_file")
            if not dem_file or not os.path.exists(dem_file):
                print(f"DEM file not found: {dem_file}")
                return None
            
            # Update reference origin from metadata if available
            if "reference_origin" in metadata:
                ref_data = metadata["reference_origin"]
                self.ref_origin = (
                    ref_data["latitude"],
                    ref_data["longitude"],
                    ref_data["altitude"]
                )
                # Re-setup transformers with new reference origin
                self._setup_transformers()
                print(f"Updated reference origin from metadata: {self.ref_origin}")
            
            print(f"Loading local topography data from: {dem_file}")
            
            # Read DEM with rasterio
            with rasterio.open(dem_file) as src:
                Z = src.read(1)
                Z = np.where(Z == src.nodata, np.nan, Z)
                transform = src.transform
                height, width = Z.shape
                
                # Create coordinate arrays
                xs = np.arange(width)
                ys = np.arange(height)
                X, Y = np.meshgrid(xs, ys)
                X, Y = rasterio.transform.xy(transform, Y, X, offset="center")
                X = np.array(X)
                Y = np.array(Y)
                
                # Ensure X and Y are 2D arrays
                if X.ndim == 1:
                    X = X.reshape(height, width)
                if Y.ndim == 1:
                    Y = Y.reshape(height, width)
                
                # Convert lat/lon grid to ENU meters for visualization
                X_enu = np.zeros_like(X)
                Y_enu = np.zeros_like(Y)
                
                # Process the transformation more efficiently
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        try:
                            x_enu, y_enu = self.transformer_llh2enu.transform(X[i, j], Y[i, j])
                            X_enu[i, j] = x_enu
                            Y_enu[i, j] = y_enu
                        except Exception as e:
                            # If transformation fails, use fallback
                            print(f"Coordinate transformation failed at ({i}, {j}): {e}")
                            # Use simple offset from reference
                            lat0, lon0, _ = self.ref_origin
                            X_enu[i, j] = (X[i, j] - lon0) * 111320 * np.cos(np.radians(lat0))
                            Y_enu[i, j] = (Y[i, j] - lat0) * 111320
            
            print(f"Local DEM shape: {Z.shape}, X_enu range: {X_enu.min()}-{X_enu.max()}, Y_enu range: {Y_enu.min()}-{Y_enu.max()}")
            print(f"Array shapes - X: {X_enu.shape}, Y: {Y_enu.shape}, Z: {Z.shape}")
            return X_enu, Y_enu, Z
            
        except Exception as e:
            print(f"Error loading local topography data: {e}")
            return None
    
    def get_dem_data(self, bounds: Tuple[float, float, float, float], 
                     resolution: int = 30) -> Optional[np.ndarray]:
        """
        Get DEM data - first tries local data, then falls back to API download.
        Args:
            bounds: (x_min, x_max, y_min, y_max) in ENU meters
            resolution: DEM resolution in meters (30, 10, 3, 1)
        Returns:
            (X, Y, Z) meshgrid in ENU meters, or None if failed
        """
        # First try to load local topography data
        local_data = self.load_local_topography_data()
        if local_data is not None:
            print("Successfully loaded local topography data")
            return local_data
        
        print("Local topography data not available, falling back to API download...")
        
        # Fallback to original API download method
        try:
            # Convert ENU bounds to lat/lon using reference origin
            x_min, x_max, y_min, y_max = bounds
            
            # Convert corners to lat/lon
            lat_sw, lon_sw = self.enu_to_latlon(x_min, y_min, 0)
            lat_ne, lon_ne = self.enu_to_latlon(x_max, y_max, 0)
            lat_nw, lon_nw = self.enu_to_latlon(x_min, y_max, 0)
            lat_se, lon_se = self.enu_to_latlon(x_max, y_min, 0)
            
            # Find bounding box in lat/lon
            lats = [lat_sw, lat_ne, lat_nw, lat_se]
            lons = [lon_sw, lon_ne, lon_nw, lon_se]
            south, north = min(lats), max(lats)
            west, east = min(lons), max(lons)
            
            # Add small buffer to ensure we get enough data
            buffer = 0.001  # ~100m buffer
            south -= buffer
            north += buffer
            west -= buffer
            east += buffer
            
            params = {
                "demtype": self.dem_type,
                "west": west,
                "east": east,
                "south": south,
                "north": north,
                "outputFormat": "GTiff",
                "API_Key": self.api_key or "demoapikeyot2022"
            }
            print(f"Requesting DEM from OpenTopography: {params}")
            resp = requests.get(self.base_url, params=params, timeout=60)
            if resp.status_code != 200 or b"<html" in resp.content[:20]:
                print(f"OpenTopography API error: {resp.status_code} {resp.text[:200]}")
                raise Exception("OpenTopography API error")
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(resp.content)
                tif_path = tmp.name
                self.last_dem_path = tif_path
            # Read DEM with rasterio
            with rasterio.open(tif_path) as src:
                Z = src.read(1)
                Z = np.where(Z == src.nodata, np.nan, Z)
                transform = src.transform
                height, width = Z.shape
                
                # Create coordinate arrays
                xs = np.arange(width)
                ys = np.arange(height)
                X, Y = np.meshgrid(xs, ys)
                X, Y = rasterio.transform.xy(transform, Y, X, offset="center")
                X = np.array(X)
                Y = np.array(Y)
                
                # Ensure X and Y are 2D arrays
                if X.ndim == 1:
                    X = X.reshape(height, width)
                if Y.ndim == 1:
                    Y = Y.reshape(height, width)
                
                # Convert lat/lon grid to ENU meters for visualization
                # Use the ENU projection directly for better performance
                X_enu = np.zeros_like(X)
                Y_enu = np.zeros_like(Y)
                
                # Process the transformation more efficiently
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        try:
                            x_enu, y_enu = self.transformer_llh2enu.transform(X[i, j], Y[i, j])
                            X_enu[i, j] = x_enu
                            Y_enu[i, j] = y_enu
                        except Exception as e:
                            # If transformation fails, use fallback
                            print(f"Coordinate transformation failed at ({i}, {j}): {e}")
                            # Use simple offset from reference
                            lat0, lon0, _ = self.ref_origin
                            X_enu[i, j] = (X[i, j] - lon0) * 111320 * np.cos(np.radians(lat0))
                            Y_enu[i, j] = (Y[i, j] - lat0) * 111320
            # Remove temp file
            os.remove(tif_path)
            print(f"DEM shape: {Z.shape}, X_enu range: {X_enu.min()}-{X_enu.max()}, Y_enu range: {Y_enu.min()}-{Y_enu.max()}")
            print(f"Array shapes - X: {X_enu.shape}, Y: {Y_enu.shape}, Z: {Z.shape}")
            return X_enu, Y_enu, Z
        except Exception as e:
            print(f"Error downloading or reading topography: {e}")
            print("Falling back to synthetic high-fidelity topography...")
            # Fallback to synthetic
            x_min, x_max, y_min, y_max = bounds
            x_coords = np.linspace(x_min, x_max, 200)
            y_coords = np.linspace(y_min, y_max, 200)
            X, Y = np.meshgrid(x_coords, y_coords)
            Z = self._generate_synthetic_topography(X, Y)
            return X, Y, Z
    
    def _generate_synthetic_topography(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Generate realistic synthetic topography with surface starting at z=0."""
        Z = np.zeros_like(X)
        # Base elevation (surface starts at z=0)
        Z += 50 * np.sin(X / 100) * np.cos(Y / 100)
        # Add mountains
        mountain_centers = [(0, 0), (50, 50), (-30, 40)]
        for cx, cy in mountain_centers:
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            Z += 200 * np.exp(-dist**2 / (2 * 30**2))
        # Add gentle valleys (but keep them above z=0)
        valley_centers = [(20, -20), (-40, -30)]
        for cx, cy in valley_centers:
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            Z -= 20 * np.exp(-dist**2 / (2 * 40**2))  # Very gentle valleys
        # Add noise for realism
        Z += np.random.normal(0, 2, X.shape)
        # Ensure minimum elevation is at z=0 (ground surface)
        Z = np.maximum(Z, 0)
        return Z

class TrajectoryVisualizer:
    """Main visualization class for trajectory display."""
    
    def __init__(self, topography_manager: TopographyManager = None, ref_origin=(0.0, 0.0, 0.0), visualization_mode='auto', topology_offset: float = 300.0):
        """Initialize the trajectory visualizer."""
        self.topography_manager = topography_manager or TopographyManager(ref_origin=ref_origin)
        self.ref_origin = ref_origin
        self.visualization_mode = visualization_mode
        self.topology_offset = topology_offset
        
        # Data storage
        self.trajectories = []
        self.protected_regions = []
        self.missiles = []
        self.radars = []
        self.collision_events = []
        self.global_time_steps = []
        
        # Visualization components
        self.plotter = None
        self.topography_mesh = None
        self.backdrop_mesh = None
        self.animation_data = []
        
        # Register cleanup
        register_cleanup()
    
    def close(self):
        """Close the visualizer and clean up resources."""
        if self.plotter is not None:
            try:
                self.plotter.close()
                self.plotter = None
            except:
                pass
        
        # Clear large data structures
        self.topography_mesh = None
        self.animation_data = []
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def load_trajectories_from_json(self, json_file: str) -> List[Trajectory]:
        """Load trajectory data from JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        trajectories = []
        for track_data in data['track_history']:
            points = []
            for point_data in track_data['timesteps']:
                point = TrajectoryPoint(
                    time=point_data['time'],
                    position=tuple(point_data['position']),
                    velocity=tuple(point_data['velocity'])
                )
                points.append(point)
            
            trajectory = Trajectory(id=track_data['id'], points=points)
            trajectories.append(trajectory)
        
        self.trajectories = trajectories
        return trajectories
    
    def load_protected_regions_from_json(self, json_file: str) -> List[ProtectedRegion]:
        """Load protected region data from JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        protected_regions = []
        if 'protected_regions' in data:
            for region_data in data['protected_regions']:
                region = ProtectedRegion(
                    id=region_data['id'],
                    centroid=tuple(region_data['centroid']),
                    radius=region_data['radius'],
                    height_limit=region_data['height_limit'],
                    name=region_data['name']
                )
                protected_regions.append(region)
        
        self.protected_regions = protected_regions
        return protected_regions
    
    def load_missiles_from_json(self, json_file: str) -> List[Missile]:
        """Load missile data from JSON file."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            missiles = []
            missile_history = data.get('missile_history', [])
            
            for missile_data in missile_history:
                missile_id = missile_data['id']
                points = []
                
                for timestep in missile_data['timesteps']:
                    point = TrajectoryPoint(
                        time=timestep['time'],
                        position=tuple(timestep['position']),
                        velocity=tuple(timestep['velocity'])
                    )
                    points.append(point)
                
                missile = Missile(id=missile_id, points=points)
                missiles.append(missile)
            
            self.missiles = missiles
            print(f"Loaded {len(missiles)} missiles")
            return missiles
            
        except Exception as e:
            print(f"Error loading missiles: {e}")
            self.missiles = []
            return []

    def load_radars_from_json(self, json_file: str) -> List[Radar]:
        """Load radar data from JSON file."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            radars = []
            resources = data.get('resources', [])
            
            for resource_data in resources:
                if resource_data.get('type') == 'radar':
                    radar = Radar(
                        id=resource_data['id'],
                        emplacement=tuple(resource_data['emplacement']),
                        type=resource_data['type'],
                        description=resource_data['description']
                    )
                    radars.append(radar)
            
            self.radars = radars
            print(f"Loaded {len(radars)} radar stations")
            return radars
            
        except Exception as e:
            print(f"Error loading radars: {e}")
            self.radars = []
            return []
    
    def detect_collisions(self, collision_threshold: float = 5.0) -> List[CollisionEvent]:
        """Detect collisions between trajectories and missiles."""
        collision_events = []
        all_objects = []
        
        # Collect all trajectory and missile objects
        for trajectory in self.trajectories:
            for point in trajectory.points:
                all_objects.append({
                    'type': 'trajectory',
                    'id': trajectory.id,
                    'time': point.time,
                    'position': point.position
                })
        
        for missile in self.missiles:
            for point in missile.points:
                all_objects.append({
                    'type': 'missile',
                    'id': missile.id,
                    'time': point.time,
                    'position': point.position
                })
        
        # Sort by time
        all_objects.sort(key=lambda x: x['time'])
        
        # Check for collisions at each time step
        for i, obj1 in enumerate(all_objects):
            for j, obj2 in enumerate(all_objects[i+1:], i+1):
                # Only check objects at the same time
                if abs(obj1['time'] - obj2['time']) < 0.1:  # Within 0.1 seconds
                    # Calculate distance between objects
                    pos1 = np.array(obj1['position'])
                    pos2 = np.array(obj2['position'])
                    distance = np.linalg.norm(pos1 - pos2)
                    
                    if distance <= collision_threshold:
                        # Check if this collision is already recorded
                        collision_exists = False
                        for event in collision_events:
                            if (abs(event.time - obj1['time']) < 0.1 and 
                                event.participants == [obj1['id'], obj2['id']]):
                                collision_exists = True
                                break
                        
                        if not collision_exists:
                            # Create collision event
                            collision_pos = tuple((pos1 + pos2) / 2)  # Midpoint
                            collision_event = CollisionEvent(
                                time=obj1['time'],
                                position=collision_pos,
                                participants=[obj1['id'], obj2['id']]
                            )
                            collision_events.append(collision_event)
        
        self.collision_events = collision_events
        print(f"Detected {len(collision_events)} collision events")
        return collision_events
    
    def create_explosion_mesh(self, position: Tuple[float, float, float], 
                            radius: float, time_factor: float = 1.0) -> pv.PolyData:
        """Create an explosion effect mesh."""
        x, y, z = position
        
        # Create a sphere for the explosion
        sphere = pv.Sphere(center=(x, y, z), radius=radius * time_factor)
        
        # Add some random particles for explosion effect
        n_particles = int(50 * time_factor)
        if n_particles > 0:
            particles = []
            for _ in range(n_particles):
                # Random direction from explosion center
                angle_xy = np.random.uniform(0, 2 * np.pi)
                angle_z = np.random.uniform(0, np.pi)
                distance = np.random.uniform(0, radius * time_factor)
                
                px = x + distance * np.sin(angle_z) * np.cos(angle_xy)
                py = y + distance * np.sin(angle_z) * np.sin(angle_xy)
                pz = z + distance * np.cos(angle_z)
                
                particles.append([px, py, pz])
            
            if particles:
                particle_mesh = pv.PolyData(particles)
                # Combine sphere and particles
                explosion_mesh = sphere.merge(particle_mesh)
                return explosion_mesh
        
        return sphere
    
    def load_reference_lla_from_json(self, json_file: str) -> Tuple[float, float, float]:
        """Load reference latitude, longitude, altitude from JSON file."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            ref_data = data.get('reference_lla', {})
            lat = ref_data.get('latitude', 0.0)
            lon = ref_data.get('longitude', 0.0)
            alt = ref_data.get('altitude', 0.0)
            description = ref_data.get('description', 'Unknown location')
            
            print(f"Loaded reference LLA: ({lat}, {lon}, {alt})")
            print(f"Reference location: {description}")
            
            # Update the reference origin in the topography manager
            self.ref_origin = (lat, lon, alt)
            if hasattr(self.topography_manager, 'ref_origin'):
                self.topography_manager.ref_origin = (lat, lon, alt)
                self.topography_manager._setup_transformers()
            
            return (lat, lon, alt)
            
        except Exception as e:
            print(f"Error loading reference LLA: {e}")
            return (0.0, 0.0, 0.0)
    
    def load_time_steps_from_json(self, json_file: str) -> List[float]:
        """Load global time steps from JSON file."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            time_steps = data.get('time_steps', [])
            if time_steps:
                print(f"Loaded {len(time_steps)} global time steps: {time_steps}")
                self.global_time_steps = time_steps
            else:
                print("No global time_steps found in JSON, will use trajectory-based timing")
                self.global_time_steps = []
            
            return time_steps
            
        except Exception as e:
            print(f"Error loading time steps: {e}")
            self.global_time_steps = []
            return []
    
    def calculate_bounds(self) -> Tuple[float, float, float, float]:
        """Calculate the bounding box for all trajectories and missiles."""
        all_positions = []
        
        # Add trajectory positions
        for trajectory in self.trajectories:
            for point in trajectory.points:
                all_positions.append(point.position)
        
        # Add missile positions
        for missile in self.missiles:
            for point in missile.points:
                all_positions.append(point.position)
        
        if not all_positions:
            return (-100, 100, -100, 100)
        
        positions = np.array(all_positions)
        x_min, y_min = positions[:, 0].min(), positions[:, 1].min()
        x_max, y_max = positions[:, 0].max(), positions[:, 1].max()
        
        # Add padding to create a reasonable bounding box for DEM data
        padding = 500  # 500 meters padding for real-world DEM data
        return (x_min - padding, x_max + padding, y_min - padding, y_max + padding)
    
    def setup_topography(self, bounds: Tuple[float, float, float, float], satellite_path: str = None, topology_offset: float = 300.0):
        """Setup topographic data for visualization with optional satellite overlay."""
        import numpy as np
        print("Setting up topography...")
        
        # If satellite imagery is provided, use a flat mesh matching the simulation bounds
        if satellite_path and os.path.exists(satellite_path):
            print("Loading satellite imagery overlay...")
            success = self.topography_manager.satellite_manager.load_satellite_imagery(satellite_path)
            if success:
                print("Satellite imagery loaded successfully")
                print("Using flat surface scaled to simulation bounds for visualization")
                import rasterio
                import numpy as np
                # Load the image and get its shape
                with rasterio.open(satellite_path) as src:
                    # Crop or resample to a reasonable size (e.g., 1000x1000)
                    height, width = src.shape
                    max_dim = 1000
                    if height > max_dim or width > max_dim:
                        # Calculate window to crop center
                        start_row = height // 2 - max_dim // 2
                        start_col = width // 2 - max_dim // 2
                        end_row = start_row + max_dim
                        end_col = start_col + max_dim
                        if src.count >= 3:
                            img = src.read([1, 2, 3], window=((start_row, end_row), (start_col, end_col)))
                        else:
                            img = src.read(1, window=((start_row, end_row), (start_col, end_col)))
                            img = np.stack([img, img, img], axis=0)
                        img_height, img_width = max_dim, max_dim
                    else:
                        if src.count >= 3:
                            img = src.read([1, 2, 3])
                        else:
                            img = src.read(1)
                            img = np.stack([img, img, img], axis=0)
                        img_height, img_width = height, width
                # Use simulation bounds for mesh
                x_min, x_max, y_min, y_max = bounds
                x_coords = np.linspace(x_min, x_max, img_width)
                y_coords = np.linspace(y_min, y_max, img_height)
                X, Y = np.meshgrid(x_coords, y_coords)
                Z = np.zeros_like(X)  # Surface at z=0
                print(f"Created flat surface with shape: {X.shape} covering bounds: {bounds} at altitude 0m")
                # Create PyVista mesh
                import pyvista as pv
                grid = pv.StructuredGrid()
                grid.points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
                grid.dimensions = [X.shape[1], X.shape[0], 1]
                # Create a surface mesh
                surface = grid.extract_surface()
                # Create normalized texture coordinates
                u = (X - X.min()) / (X.max() - X.min())
                v = (Y - Y.min()) / (Y.max() - Y.min())
                tex_coords = np.column_stack([u.flatten(), v.flatten()])
                surface.active_texture_coordinates = tex_coords.astype(np.float32)
                print(f"Set texture coordinates on flat surface")
                print(f"Texture coordinates range: {tex_coords.min()} to {tex_coords.max()}")
                self.topography_mesh = surface
                return
            else:
                print("Failed to load satellite imagery, using topography only")
        # Otherwise, use the original topography pipeline
        topo_data = self.topography_manager.get_dem_data(bounds)
        if topo_data is None:
            print("Failed to get topography, using flat surface")
            return
        X, Y, Z = topo_data
        if X.ndim != 2 or Y.ndim != 2 or Z.ndim != 2:
            print("Error: Topography arrays must be 2D")
            return
        if X.shape != Y.shape or Y.shape != Z.shape:
            print("Error: Topography arrays must have the same shape")
            return
        
        # Create a flat plane that extends forever as a backdrop
        # Calculate extended bounds to ensure the plane covers the entire view
        x_min, x_max, y_min, y_max = bounds
        padding = max(x_max - x_min, y_max - y_min) * 3.0  # Large padding to extend beyond view
        
        # Create extended coordinates for a much larger plane
        x_min_extended = x_min - padding
        x_max_extended = x_max + padding
        y_min_extended = y_min - padding
        y_max_extended = y_max + padding
        
        # Use the topology offset to position the plane
        surface_level = topology_offset
        z_level = -surface_level  # Plane at negative z to be below objects at z=0
        
        print(f"Creating flat tan ground plane extending from ({x_min_extended:.1f}, {y_min_extended:.1f}) to ({x_max_extended:.1f}, {y_max_extended:.1f})")
        print(f"Ground plane at z={z_level:.1f}m")
        print(f"Objects at z=0 will appear {surface_level:.1f}m above the ground plane")
        
        # Create a simple large rectangular plane as backdrop
        import pyvista as pv
        backdrop_plane = pv.Plane(
            center=(0, 0, z_level),
            direction=(0, 0, 1),
            i_size=x_max_extended - x_min_extended,
            j_size=y_max_extended - y_min_extended,
            i_resolution=100,
            j_resolution=100
        )
        
        # Create the actual topography mesh
        print(f"Creating topography mesh with shape: {X.shape}")
        print(f"Original topography elevation range: {Z.min():.1f}m to {Z.max():.1f}m")
        
        # Shift the topography down using the provided topology offset
        Z = Z - surface_level
        print(f"Shifting topography down by {surface_level}m")
        print(f"Adjusted topography elevation range: {Z.min():.1f}m to {Z.max():.1f}m")
        
        # Create topography mesh
        grid = pv.StructuredGrid()
        grid.points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        grid.dimensions = [X.shape[1], X.shape[0], 1]
        grid.point_data['elevation'] = Z.flatten()
        
        # Create surface mesh from grid
        surface = grid.extract_surface()
        surface = surface.smooth(n_iter=10, relaxation_factor=0.1)
        surface.point_data['elevation'] = Z.flatten()
        
        # Store both meshes - we'll render the backdrop first, then the topography on top
        self.backdrop_mesh = backdrop_plane
        self.topography_mesh = surface
    
    def create_trajectory_meshes(self) -> Dict[int, pv.PolyData]:
        """Create PyVista meshes for all trajectories."""
        trajectory_meshes = {}
        
        for trajectory in self.trajectories:
            if not trajectory.points:
                continue
            
            # Extract positions and velocities
            positions = np.array([point.position for point in trajectory.points])
            velocities = np.array([point.velocity for point in trajectory.points])
            times = np.array([point.time for point in trajectory.points])
            
            # Create line for trajectory path
            line = pv.lines_from_points(positions)
            line.point_data['time'] = times
            line.point_data['velocity_magnitude'] = np.linalg.norm(velocities, axis=1)
            
            # Create velocity vectors - handle each point individually
            velocity_vectors = []
            for i, (pos, vel) in enumerate(zip(positions, velocities)):
                if np.linalg.norm(vel) > 0:  # Only create arrow if velocity is non-zero
                    # Normalize velocity for direction
                    vel_norm = vel / np.linalg.norm(vel)
                    arrow = pv.Arrow(
                        start=pos,
                        direction=vel_norm,
                        scale=2.0,
                        tip_length=0.3,
                        tip_radius=0.1,
                        shaft_radius=0.05
                    )
                    velocity_vectors.append(arrow)
            
            # Combine all velocity vectors if any exist
            if velocity_vectors:
                combined_vectors = velocity_vectors[0]
                for vec in velocity_vectors[1:]:
                    combined_vectors = combined_vectors.merge(vec)
            else:
                # Create empty mesh if no velocity vectors
                combined_vectors = pv.PolyData()
            
            trajectory_meshes[trajectory.id] = {
                'line': line,
                'velocity_vectors': combined_vectors,
                'positions': positions,
                'velocities': velocities,
                'times': times
            }
        
        return trajectory_meshes
    
    def create_missile_meshes(self) -> Dict[int, pv.PolyData]:
        """Create PyVista meshes for all missiles."""
        missile_meshes = {}
        
        for missile in self.missiles:
            if not missile.points:
                continue
            
            # Extract positions and velocities
            positions = np.array([point.position for point in missile.points])
            velocities = np.array([point.velocity for point in missile.points])
            times = np.array([point.time for point in missile.points])
            
            # Create line for missile path
            line = pv.lines_from_points(positions)
            line.point_data['time'] = times
            line.point_data['velocity_magnitude'] = np.linalg.norm(velocities, axis=1)
            
            # Create velocity vectors - handle each point individually
            velocity_vectors = []
            for i, (pos, vel) in enumerate(zip(positions, velocities)):
                if np.linalg.norm(vel) > 0:  # Only create arrow if velocity is non-zero
                    # Normalize velocity for direction
                    vel_norm = vel / np.linalg.norm(vel)
                    arrow = pv.Arrow(
                        start=pos,
                        direction=vel_norm,
                        scale=2.0,
                        tip_length=0.3,
                        tip_radius=0.1,
                        shaft_radius=0.05
                    )
                    velocity_vectors.append(arrow)
            
            # Combine all velocity vectors if any exist
            if velocity_vectors:
                combined_vectors = velocity_vectors[0]
                for vec in velocity_vectors[1:]:
                    combined_vectors = combined_vectors.merge(vec)
            else:
                # Create empty mesh if no velocity vectors
                combined_vectors = pv.PolyData()
            
            missile_meshes[missile.id] = {
                'line': line,
                'velocity_vectors': combined_vectors,
                'positions': positions,
                'velocities': velocities,
                'times': times
            }
        
        return missile_meshes
    
    def create_protected_region_meshes(self) -> Dict[int, pv.PolyData]:
        """Create PyVista cylindrical meshes for all protected regions."""
        protected_region_meshes = {}
        
        for region in self.protected_regions:
            # Create cylinder mesh
            # PyVista cylinder is created along z-axis, so we need to position it correctly
            cylinder = pv.Cylinder(
                center=(region.centroid[0], region.centroid[1], region.height_limit / 2),
                direction=(0, 0, 1),
                radius=region.radius,
                height=region.height_limit,
                resolution=32  # Number of points around circumference
            )
            
            protected_region_meshes[region.id] = {
                'mesh': cylinder,
                'region': region
            }
        
        return protected_region_meshes
    
    def setup_visualization(self, window_size: Tuple[int, int] = (1920, 1080), off_screen: bool = False):
        """Setup the main visualization window."""
        # Check if running in WSLg and adjust window size accordingly
        is_wslg = False
        try:
            with open('/proc/version', 'r') as f:
                content = f.read().lower()
                if 'microsoft' in content and 'wslg' in content:
                    is_wslg = True
        except:
            pass
        
        if is_wslg:
            print("Using WSLg-optimized window size: 1280x720")
            window_size = (1280, 720)
        else:
            window_size = window_size
        
        # Create plotter with WSLg-optimized settings
        if is_wslg:
            # WSLg provides excellent graphics support
            self.plotter = pv.Plotter(
                off_screen=off_screen, 
                window_size=window_size,
                lighting='three lights'  # Use standard lighting
            )
        else:
            self.plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
        
        # Register plotter for cleanup
        register_cleanup(self.plotter)
        
        # Set background to tan
        self.plotter.set_background('tan')
        
        # Add coordinate axes
        self.plotter.add_axes(
            xlabel='East (m)', 
            ylabel='North (m)', 
            zlabel='Up (m)',
            line_width=2,
            labels_off=False
        )
        
        # Set camera position to center on trajectories
        if self.trajectories:
            # Calculate center of all trajectory points
            all_positions = []
            for trajectory in self.trajectories:
                for point in trajectory.points:
                    all_positions.append(point.position)
            
            if all_positions:
                positions = np.array(all_positions)
                center = positions.mean(axis=0)
                
                # Calculate bounds for camera distance
                bounds = self.calculate_bounds()
                x_range = bounds[1] - bounds[0]
                y_range = bounds[3] - bounds[2]
                max_range = max(x_range, y_range)
                
                # Set camera position to look at center with appropriate distance
                camera_distance = max_range * 1.5  # Adjust multiplier as needed
                self.plotter.camera_position = [
                    (center[0] + camera_distance, center[1] + camera_distance, center[2] + camera_distance),
                    center,
                    (0, 0, 1)  # Up vector
                ]
                
                # Zoom to fit all data
                self.plotter.camera.zoom(0.8)
        else:
            # Fallback to iso view if no trajectories
            self.plotter.camera_position = 'iso'
            self.plotter.camera.zoom(1.5)
    
    def add_topography_to_plot(self):
        """Add topography to the visualization with optional satellite overlay."""
        if self.topography_mesh is None:
            return
        
        # First, add the backdrop plane if it exists
        if hasattr(self, 'backdrop_mesh') and self.backdrop_mesh is not None:
            print("Adding flat tan backdrop plane...")
            self.plotter.add_mesh(
                self.backdrop_mesh,
                color='tan',  # Solid tan/sand color for backdrop
                show_edges=False,
                lighting=True,
                ambient=0.3,
                diffuse=0.7,
                specular=0.2,
                specular_power=15,
                smooth_shading=True,
                opacity=0.9
            )
        
        # Check if satellite texture is available
        if self.topography_manager.satellite_manager.satellite_texture is not None:
            print("Adding topography with satellite imagery overlay...")
            # Add topography mesh with satellite texture
            # Disable lighting to make satellite texture more visible
            self.plotter.add_mesh(
                self.topography_mesh,
                texture=self.topography_manager.satellite_manager.satellite_texture,
                show_edges=False,
                lighting=False,  # Disable lighting for better texture visibility
                opacity=1.0  # Full opacity for satellite imagery
            )
        else:
            print("Adding topography with solid tan color...")
            # Use solid tan color for topography to match the backdrop
            self.plotter.add_mesh(
                self.topography_mesh,
                color='tan',  # Solid tan/sand color to match backdrop
                show_edges=False,
                lighting=True,
                ambient=0.3,
                diffuse=0.7,
                specular=0.2,
                specular_power=15,
                smooth_shading=True,
                opacity=0.9
            )
    
    def add_trajectories_to_plot(self, trajectory_meshes: Dict[int, pv.PolyData]):
        """Add trajectory meshes to the visualization."""
        # Use blue color for all trajectories
        color = 'blue'
        
        for trajectory_id, mesh_data in trajectory_meshes.items():
            # Add trajectory line as thick tube for better visibility (same size as missiles)
            tube = mesh_data['line'].tube(radius=15.0)  # Same as missiles
            self.plotter.add_mesh(
                tube,
                color=color,
                line_width=30,  # Same as missiles
                show_scalar_bar=False
            )
            
            # Add velocity vectors
            if mesh_data['positions'].shape[0] > 0:
                # Only add velocity vectors if the mesh has points
                if mesh_data['velocity_vectors'].n_points > 0:
                    self.plotter.add_mesh(
                        mesh_data['velocity_vectors'],
                        color=color,
                        opacity=1.0  # Increased from 0.8 to 1.0 (30% more opaque)
                    )
            
            # Add trajectory ID label
            if mesh_data['positions'].shape[0] > 0:
                start_pos = mesh_data['positions'][0]
                self.plotter.add_point_labels(
                    [start_pos],
                    [f'Trajectory {trajectory_id}'],
                    font_size=16,
                    bold=True,
                    text_color=color
                )
    
    def add_missiles_to_plot(self, missile_meshes: Dict[int, pv.PolyData]):
        """Add missile meshes to the visualization."""
        # All missiles use yellow color
        missile_color = 'yellow'
        
        for i, (missile_id, mesh_data) in enumerate(missile_meshes.items()):
            # Add missile line as thick tube for better visibility
            tube = mesh_data['line'].tube(radius=15.0)  # Increased from 8.0 to 15.0
            self.plotter.add_mesh(
                tube,
                color=missile_color,
                line_width=30,  # Increased from 20 to 30
                show_scalar_bar=False,
                lighting=True,
                ambient=0.8,
                diffuse=0.2,
                specular=0.5,
                specular_power=10
            )
            
            # Add velocity vectors
            if mesh_data['positions'].shape[0] > 0:
                # Only add velocity vectors if the mesh has points
                if mesh_data['velocity_vectors'].n_points > 0:
                    self.plotter.add_mesh(
                        mesh_data['velocity_vectors'],
                        color=missile_color,
                        opacity=1.0  # Increased from 0.8 to 1.0 (30% more opaque)
                    )
            
            # Add missile ID label
            if mesh_data['positions'].shape[0] > 0:
                start_pos = mesh_data['positions'][0]
                self.plotter.add_point_labels(
                    [start_pos],
                    [f'Missile {missile_id}'],
                    font_size=16,
                    bold=True,
                    text_color=missile_color
                )
    
    def create_animation_data(self, trajectory_meshes: Dict[int, pv.PolyData], 
                            missile_meshes: Dict[int, pv.PolyData] = None,
                            intermediate_frames: int = 4):
        """Prepare data for animation with trails and interpolation."""
        self.animation_data = []
        
        # Use global time steps if available, otherwise extract from trajectories
        if self.global_time_steps:
            print(f"Using global time steps for animation: {self.global_time_steps}")
            # Create intermediate frames for smoother animation
            all_times = set()
            for i in range(len(self.global_time_steps) - 1):
                t1, t2 = self.global_time_steps[i], self.global_time_steps[i + 1]
                # Add intermediate frames between each global time step
                for j in range(intermediate_frames + 1):
                    intermediate_time = t1 + j * (t2 - t1) / intermediate_frames
                    all_times.add(intermediate_time)
            # Add the last global time step
            all_times.add(self.global_time_steps[-1])
        else:
            print("Using trajectory-based time steps for animation")
            # Get all unique timestamps from trajectories and missiles
            all_times = set()
            for mesh_data in trajectory_meshes.values():
                all_times.update(mesh_data['times'])
            
            if missile_meshes:
                for mesh_data in missile_meshes.values():
                    all_times.update(mesh_data['times'])
        
        # Add collision event times
        for event in self.collision_events:
            all_times.add(event.time)
            # Add explosion animation times
            for t in np.arange(event.time, event.time + event.explosion_duration, 0.1):
                all_times.add(t)
        
        sorted_times = sorted(all_times)
        print(f"Animation will have {len(sorted_times)} frames at times: {sorted_times}")
        
        # Create trajectory and missile data structures for interpolation
        trajectory_data = {}
        missile_data = {}
        
        # Process trajectory data
        for trajectory_id, mesh_data in trajectory_meshes.items():
            times = mesh_data['times']
            positions = mesh_data['positions']
            velocities = mesh_data['velocities']
            
            trajectory_data[trajectory_id] = {
                'times': times,
                'positions': positions,
                'velocities': velocities,
                'trail_positions': [],  # Will be built progressively
                'trail_velocities': []  # Will be built progressively
            }
        
        # Process missile data
        if missile_meshes:
            for missile_id, mesh_data in missile_meshes.items():
                times = mesh_data['times']
                positions = mesh_data['positions']
                velocities = mesh_data['velocities']
                
                missile_data[missile_id] = {
                    'times': times,
                    'positions': positions,
                    'velocities': velocities,
                    'trail_positions': [],  # Will be built progressively
                    'trail_velocities': []  # Will be built progressively
                }
        
        for i, time in enumerate(sorted_times):
            frame_data = {
                'time': time, 
                'active_trajectories': [], 
                'active_missiles': [],
                'explosions': []
            }
            
            # Process trajectory data for this frame
            for trajectory_id, data in trajectory_data.items():
                # Find the current position and velocity at this time
                current_pos = None
                current_vel = None
                
                # Check if we have exact time match
                time_indices = np.where(np.isclose(data['times'], time))[0]
                if len(time_indices) > 0:
                    current_pos = data['positions'][time_indices[0]]
                    current_vel = data['velocities'][time_indices[0]]
                else:
                    # Interpolate between waypoints
                    current_pos, current_vel = self._interpolate_position_velocity(
                        data['times'], data['positions'], data['velocities'], time
                    )
                
                if current_pos is not None:
                    # Add to trail
                    data['trail_positions'].append(current_pos)
                    data['trail_velocities'].append(current_vel)
                    
                    frame_data['active_trajectories'].append({
                        'id': trajectory_id,
                        'current_position': current_pos,
                        'current_velocity': current_vel,
                        'trail_positions': data['trail_positions'].copy(),
                        'trail_velocities': data['trail_velocities'].copy()
                    })
            
            # Process missile data for this frame
            for missile_id, data in missile_data.items():
                # Find the current position and velocity at this time
                current_pos = None
                current_vel = None
                
                # Check if we have exact time match
                time_indices = np.where(np.isclose(data['times'], time))[0]
                if len(time_indices) > 0:
                    current_pos = data['positions'][time_indices[0]]
                    current_vel = data['velocities'][time_indices[0]]
                else:
                    # Interpolate between waypoints
                    current_pos, current_vel = self._interpolate_position_velocity(
                        data['times'], data['positions'], data['velocities'], time
                    )
                
                if current_pos is not None:
                    # Add to trail
                    data['trail_positions'].append(current_pos)
                    data['trail_velocities'].append(current_vel)
                    
                    frame_data['active_missiles'].append({
                        'id': missile_id,
                        'current_position': current_pos,
                        'current_velocity': current_vel,
                        'trail_positions': data['trail_positions'].copy(),
                        'trail_velocities': data['trail_velocities'].copy()
                    })
            
            # Add explosion effects for this time
            for event in self.collision_events:
                if event.time <= time <= event.time + event.explosion_duration:
                    # Calculate explosion animation factor (0 to 1)
                    explosion_progress = (time - event.time) / event.explosion_duration
                    # Use a bell curve for explosion effect (starts small, grows, then fades)
                    time_factor = 4 * explosion_progress * (1 - explosion_progress)
                    
                    frame_data['explosions'].append({
                        'position': event.position,
                        'radius': event.explosion_radius,
                        'time_factor': time_factor,
                        'participants': event.participants
                    })
            
            self.animation_data.append(frame_data)
    
    def _interpolate_position_velocity(self, times, positions, velocities, target_time):
        """Interpolate position and velocity between waypoints with proper speed scaling."""
        if len(times) == 0: 
            return None, None
        
        # Find the two waypoints to interpolate between
        if target_time <= times[0]:
            return positions[0], velocities[0]
        elif target_time >= times[-1]:
            return positions[-1], velocities[-1]
        
        # Find the indices of the surrounding waypoints
        for i in range(len(times) - 1):
            if times[i] <= target_time <= times[i + 1]:
                t1, t2 = times[i], times[i + 1]
                pos1, pos2 = positions[i], positions[i + 1]
                vel1, vel2 = velocities[i], velocities[i + 1]
                
                # Calculate the actual distance between waypoints
                distance = np.linalg.norm(pos2 - pos1)
                time_interval = t2 - t1
                
                # Calculate the required speed to travel the distance in the time interval
                required_speed = distance / time_interval if time_interval > 0 else 0
                
                # Calculate interpolation factor
                alpha = (target_time - t1) / (t2 - t1)
                
                # Interpolate position linearly
                interpolated_pos = pos1 + alpha * (pos2 - pos1)
                
                # Calculate velocity based on the direction and required speed
                if distance > 0:
                    direction = (pos2 - pos1) / distance
                    interpolated_vel = direction * required_speed
                else:
                    # If waypoints are at the same position, use average velocity
                    interpolated_vel = (vel1 + vel2) / 2
                
                return interpolated_pos, interpolated_vel
        
        return None, None
    
    def animate_trajectories(self, trajectory_meshes: Dict[int, pv.PolyData], 
                           missile_meshes: Dict[int, pv.PolyData] = None,
                           fps: int = 10, save_path: str = None, video_zoom: float = 1.0,
                           intermediate_frames: int = 4):
        """Create an animation of the trajectories and missiles."""
        if not self.animation_data:
            self.create_animation_data(trajectory_meshes, missile_meshes, intermediate_frames)
        
        print(f"Creating animation with {len(self.animation_data)} frames...")
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_paths = []
            
            # Use off_screen plotter for animation
            self.setup_visualization(off_screen=True)
            
            # Add static elements once (topography, protected regions, collision markers)
            print("Adding static elements (topography, protected regions, collision markers)...")
            self.add_topography_to_plot()
            
            # Create protected region meshes for animation
            protected_region_meshes = {}
            if self.protected_regions:
                protected_region_meshes = self.create_protected_region_meshes()
                self.add_protected_regions_to_plot(protected_region_meshes)
            
            # Add collision markers (static throughout animation)
            self.add_collision_markers_to_plot()
            
            # Calculate optimal camera position to include all elements
            self._setup_animation_camera(video_zoom)
            
            # Track dynamic actors to clear each frame
            dynamic_actors = []
            
            for i, frame_data in enumerate(self.animation_data):
                # Clear only dynamic actors from previous frame
                for actor in dynamic_actors:
                    self.plotter.remove_actor(actor)
                dynamic_actors.clear()
                
                # Add coordinate axes (dynamic - may change each frame)
                axes_actor = self.plotter.add_axes(
                    xlabel='East (m)', 
                    ylabel='North (m)', 
                    zlabel='Up (m)',
                    line_width=2,
                    labels_off=False
                )
                dynamic_actors.append(axes_actor)
                
                # Add time display (dynamic - changes each frame)
                time_actor = self.plotter.add_text(
                    f"Time: {frame_data['time']:.2f}s",
                    position='upper_left',
                    font_size=20,
                    color='white'
                )
                dynamic_actors.append(time_actor)
                
                # Add active trajectories for this frame
                for trajectory_info in frame_data['active_trajectories']:
                    trajectory_id = trajectory_info['id']
                    current_position = trajectory_info['current_position']
                    current_velocity = trajectory_info['current_velocity']
                    trail_positions = trajectory_info['trail_positions']
                    
                    # Check if this trajectory has been intercepted
                    trajectory_intercepted = False
                    for collision in self.collision_events:
                        if trajectory_id in collision.participants and frame_data['time'] >= collision.time:
                            trajectory_intercepted = True
                            break
                    
                    # Choose color based on interception status
                    if trajectory_intercepted:
                        color = 'green'  # Intercepted trajectories are green
                    else:
                        color = 'blue'   # Normal trajectories are blue
                    
                    # Create trail line
                    if len(trail_positions) > 1:
                        trail_points = np.array(trail_positions)
                        trail_line = pv.lines_from_points(trail_points)
                        
                        # Add trail with fading opacity
                        trail_actor = self.plotter.add_mesh(
                            trail_line,
                            color=color,
                            line_width=8,  # Same as missiles
                            opacity=0.8,
                            render_lines_as_tubes=True
                        )
                        dynamic_actors.append(trail_actor)
                    
                    # Add velocity vector (arrow head) for current position
                    if np.linalg.norm(current_velocity) > 0:
                        vel_norm = current_velocity / np.linalg.norm(current_velocity)
                        arrow = pv.Arrow(
                            start=current_position,
                            direction=vel_norm,
                            scale=12.0,  # Same as missiles
                            tip_length=0.4,
                            tip_radius=0.3,
                            shaft_radius=0.15
                        )
                        
                        vector_actor = self.plotter.add_mesh(
                            arrow,
                            color=color,
                            opacity=0.9
                        )
                        dynamic_actors.append(vector_actor)
                
                # Add active missiles for this frame
                for missile_info in frame_data['active_missiles']:
                    missile_id = missile_info['id']
                    current_position = missile_info['current_position']
                    current_velocity = missile_info['current_velocity']
                    trail_positions = missile_info['trail_positions']
                    
                    # All missiles use yellow color
                    color = 'yellow'
                    
                    # Create trail line for missile
                    if len(trail_positions) > 1:
                        trail_points = np.array(trail_positions)
                        trail_line = pv.lines_from_points(trail_points)
                        
                        # Add missile trail with fading opacity
                        trail_actor = self.plotter.add_mesh(
                            trail_line,
                            color=color,
                            line_width=8,
                            opacity=0.8,
                            render_lines_as_tubes=True
                        )
                        dynamic_actors.append(trail_actor)
                    
                    # Add velocity vector (arrow head) for current missile position
                    if np.linalg.norm(current_velocity) > 0:
                        vel_norm = current_velocity / np.linalg.norm(current_velocity)
                        arrow = pv.Arrow(
                            start=current_position,
                            direction=vel_norm,
                            scale=12.0,  # Same as trajectories
                            tip_length=0.4,
                            tip_radius=0.3,
                            shaft_radius=0.15
                        )
                        
                        vector_actor = self.plotter.add_mesh(
                            arrow,
                            color=color,
                            opacity=0.9
                        )
                        dynamic_actors.append(vector_actor)
                
                # Add explosion effects for this frame
                for explosion_info in frame_data['explosions']:
                    position = explosion_info['position']
                    radius = explosion_info['radius']
                    time_factor = explosion_info['time_factor']
                    participants = explosion_info['participants']
                    
                    # Create explosion mesh
                    explosion_mesh = self.create_explosion_mesh(position, radius, time_factor)
                    
                    # Add explosion to plot with dynamic color and opacity
                    explosion_color = 'yellow' if time_factor > 0.5 else 'orange'
                    explosion_opacity = min(0.9, time_factor * 2)  # Fade out over time
                    
                    explosion_actor = self.plotter.add_mesh(
                        explosion_mesh,
                        color=explosion_color,
                        opacity=explosion_opacity,
                        show_edges=False,
                        lighting=True,
                        ambient=0.8,
                        diffuse=0.2
                    )
                    dynamic_actors.append(explosion_actor)
                
                # Add legend (dynamic - may change each frame)
                legend_text = f"Animation Frame {i+1}/{len(self.animation_data)}\nBlue: Trajectories\nYellow: Missiles"
                if self.collision_events:
                    legend_text += "\nGreen: Intercepted Objects"
                if self.protected_regions:
                    legend_text += "\nRed Cylinders: Protected Regions"
                if self.collision_events:
                    legend_text += "\nRed X: Collision Points"
                
                # Add end position arrows in the final frame
                if i == len(self.animation_data) - 1:
                    legend_text += "\nLarge Arrows: End Positions"
                    # Add end position arrows for trajectories
                    for trajectory_info in frame_data['active_trajectories']:
                        trajectory_id = trajectory_info['id']
                        current_position = trajectory_info['current_position']
                        current_velocity = trajectory_info['current_velocity']
                        
                        # Check if this trajectory has been intercepted
                        trajectory_intercepted = False
                        for collision in self.collision_events:
                            if trajectory_id in collision.participants and frame_data['time'] >= collision.time:
                                trajectory_intercepted = True
                                break
                        
                        if np.linalg.norm(current_velocity) > 0:
                            vel_norm = current_velocity / np.linalg.norm(current_velocity)
                            arrow = pv.Arrow(
                                start=current_position,
                                direction=vel_norm,
                                scale=40.0,  # Much larger arrow for end position
                                tip_length=0.4,
                                tip_radius=0.5,
                                shaft_radius=0.25
                            )
                            
                            # Choose color based on interception status
                            if trajectory_intercepted:
                                color = 'green'  # Intercepted trajectories are green
                            else:
                                color = 'blue'   # Normal trajectories are blue
                            
                            end_arrow_actor = self.plotter.add_mesh(
                                arrow,
                                color=color,
                                opacity=0.9,
                                lighting=True,
                                ambient=0.8,
                                diffuse=0.2,
                                specular=0.5,
                                specular_power=10
                            )
                            dynamic_actors.append(end_arrow_actor)
                    
                    # Add end position arrows for missiles
                    for missile_id, mesh_data in missile_meshes.items():
                        if mesh_data['positions'].shape[0] > 0:
                            end_position = mesh_data['positions'][-1]
                            end_velocity = mesh_data['velocities'][-1]
                            
                            if np.linalg.norm(end_velocity) > 0:
                                vel_norm = end_velocity / np.linalg.norm(end_velocity)
                                arrow = pv.Arrow(
                                    start=end_position,
                                    direction=vel_norm,
                                    scale=50.0,  # Much larger arrow for missile end position
                                    tip_length=0.4,
                                    tip_radius=0.6,
                                    shaft_radius=0.3
                                )
                                
                                # All missiles use yellow color
                                color = 'yellow'
                                self.plotter.add_mesh(
                                    arrow,
                                    color=color,
                                    opacity=0.9,
                                    lighting=True,
                                    ambient=0.8,
                                    diffuse=0.2,
                                    specular=0.5,
                                    specular_power=10
                                )
                                
                                # Add end position label
                                self.plotter.add_point_labels(
                                    [end_position],
                                    [f'End M{missile_id}'],
                                    font_size=14,
                                    bold=True,
                                    text_color=color
                                )
                
                legend_actor = self.plotter.add_text(
                    legend_text,
                    position='upper_right',
                    font_size=16,
                    color='white'
                )
                dynamic_actors.append(legend_actor)
                
                # Save frame
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                self.plotter.screenshot(frame_path, window_size=(1920, 1080))
                frame_paths.append(frame_path)
                
                # Progress indicator
                if (i + 1) % 10 == 0 or i == len(self.animation_data) - 1:
                    print(f"Animation progress: {i+1}/{len(self.animation_data)} frames")
            
            # Create video from frames
            if save_path:
                print("Creating video from frames...")
                self._create_video_from_frames(frame_paths, save_path, fps)
                print(f"Animation saved to: {save_path}")
    
    def _setup_animation_camera(self, video_zoom: float = 1.0):
        """Setup camera position optimized for animation to include all elements."""
        # Calculate bounds including trajectories, protected regions, and topography
        all_points = []
        
        # Add trajectory points
        for trajectory in self.trajectories:
            for point in trajectory.points:
                all_points.append(point.position)
        
        # Add missile points
        for missile in self.missiles:
            for point in missile.points:
                all_points.append(point.position)
        
        # Add protected region points (centroids and top points)
        for region in self.protected_regions:
            all_points.append(region.centroid)
            # Add top point of cylinder
            top_point = (region.centroid[0], region.centroid[1], region.height_limit)
            all_points.append(top_point)
            # Add points around the cylinder perimeter at different heights
            for angle in np.linspace(0, 2*np.pi, 8):
                x = region.centroid[0] + region.radius * np.cos(angle)
                y = region.centroid[1] + region.radius * np.sin(angle)
                all_points.append((x, y, 0))  # Base
                all_points.append((x, y, region.height_limit))  # Top
        
        if all_points:
            points_array = np.array(all_points)
            center = points_array.mean(axis=0)
            
            # Calculate bounds
            x_min, y_min, z_min = points_array.min(axis=0)
            x_max, y_max, z_max = points_array.max(axis=0)
            
            # Calculate ranges
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            max_range = max(x_range, y_range, z_range)
            
            # Set camera position with zoom factor applied
            camera_distance = max_range * 0.5 / video_zoom  # Apply zoom factor
            
            # Position camera at an angle to show 3D perspective
            camera_pos = (
                center[0] + camera_distance * 0.7,
                center[1] + camera_distance * 0.7,
                center[2] + camera_distance * 0.5
            )
            
            self.plotter.camera_position = [
                camera_pos,
                center,
                (0, 0, 1)  # Up vector
            ]
            
            # Apply additional zoom factor
            self.plotter.camera.zoom(video_zoom)
            print(f"Animation camera set with zoom factor: {video_zoom}")
        else:
            # Fallback camera position
            self.plotter.camera_position = 'iso'
            self.plotter.camera.zoom(video_zoom)
    
    def _create_video_from_frames(self, frame_paths: List[str], output_path: str, fps: int):
        """Create video from frame images using ffmpeg."""
        try:
            import subprocess
            
            # Create ffmpeg command
            cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-framerate', str(fps),
                '-i', os.path.join(os.path.dirname(frame_paths[0]), 'frame_%04d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',  # High quality
                output_path
            ]
            
            subprocess.run(cmd, check=True)
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ffmpeg not found or failed. Please install ffmpeg to create videos.")
            print("Individual frames are available in the temporary directory.")
    
    def _setup_camera_with_zoom(self, zoom_factor: float = 1.0):
        """Setup camera position with zoom factor."""
        if self.topography_mesh is None:
            return
        
        # Calculate bounds from topography mesh
        bounds = self.topography_mesh.bounds
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        
        # Calculate center and size
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center_z = (z_min + z_max) / 2
        
        # Calculate size for camera positioning
        size_x = x_max - x_min
        size_y = y_max - y_min
        size_z = z_max - z_min
        max_size = max(size_x, size_y, size_z)
        
        # Apply zoom factor (higher zoom = closer camera)
        camera_distance = max_size * 2.0 / zoom_factor
        
        # Set camera position
        camera_pos = (center_x, center_y - camera_distance, center_z + camera_distance * 0.5)
        focal_point = (center_x, center_y, center_z)
        
        self.plotter.camera_position = [camera_pos, focal_point, (0, 0, 1)]
        self.plotter.camera.zoom(zoom_factor)
        
        print(f"Camera set with zoom factor: {zoom_factor}")
    
    def create_static_visualization(self, save_path: str = None, satellite_path: str = None, photo_zoom: float = 1.0):
        """Create a static visualization of all trajectories with optional satellite overlay."""
        if not self.trajectories:
            print("No trajectories loaded!")
            return
        
        # Calculate bounds
        bounds = self.calculate_bounds()
        
        # Setup topography with optional satellite overlay
        self.setup_topography(bounds, satellite_path, self.topology_offset)
        
        # Setup visualization (off_screen if saving)
        self.setup_visualization(off_screen=bool(save_path))
        
        # Add topography
        self.add_topography_to_plot()
        
        # Create trajectory meshes
        trajectory_meshes = self.create_trajectory_meshes()
        
        # Add trajectories
        self.add_trajectories_to_plot(trajectory_meshes)
        
        # Create and add missile meshes
        missile_meshes = None
        if self.missiles:
            missile_meshes = self.create_missile_meshes()
            self.add_missiles_to_plot(missile_meshes)
        
        # Add end position arrows for trajectories and missiles
        self.add_end_position_arrows(trajectory_meshes, missile_meshes)
        
        # Add radar stations
        self.add_radars_to_plot()
        
        # Create and add protected region meshes
        if self.protected_regions:
            protected_region_meshes = self.create_protected_region_meshes()
            self.add_protected_regions_to_plot(protected_region_meshes)
        
        # Add collision markers
        self.add_collision_markers_to_plot()
        
        # Setup camera with zoom
        self._setup_camera_with_zoom(photo_zoom)
        
        # Add legend
        legend_text = "Trajectory Visualization\nBlue: Trajectories\nYellow: Missiles"
        if self.collision_events:
            legend_text += "\nGreen: Intercepted Objects"
        if self.radars:
            legend_text += "\nPurple: Radar Stations"
        if self.protected_regions:
            legend_text += "\nRed Cylinders: Protected Regions"
        if self.collision_events:
            legend_text += "\nRed X: Collision Points"
        legend_text += "\nLarge Arrows: End Positions"
        if satellite_path:
            legend_text += "\nSatellite imagery overlay enabled"
        
        self.plotter.add_text(
            legend_text,
            position='upper_right',
            font_size=16,
            color='white'
        )
        
        # Show or save
        if save_path:
            self.plotter.screenshot(save_path, window_size=(1920, 1080))
            print(f"Static visualization saved to: {save_path}")
        else:
            self.plotter.show()
    
    def create_interactive_visualization(self, satellite_path: str = None):
        """Create an interactive visualization with optional satellite overlay."""
        if not self.trajectories:
            print("No trajectories loaded!")
            return
        
        # Calculate bounds
        bounds = self.calculate_bounds()
        
        # Setup topography with optional satellite overlay
        self.setup_topography(bounds, satellite_path, self.topology_offset)
        
        # Setup visualization
        self.setup_visualization()
        
        # Add topography
        self.add_topography_to_plot()
        
        # Create trajectory meshes
        trajectory_meshes = self.create_trajectory_meshes()
        
        # Add trajectories
        self.add_trajectories_to_plot(trajectory_meshes)
        
        # Create and add missile meshes
        missile_meshes = None
        if self.missiles:
            missile_meshes = self.create_missile_meshes()
            self.add_missiles_to_plot(missile_meshes)
        
        # Add end position arrows for trajectories and missiles
        self.add_end_position_arrows(trajectory_meshes, missile_meshes)
        
        # Add radar stations
        self.add_radars_to_plot()
        
        # Create and add protected region meshes
        if self.protected_regions:
            protected_region_meshes = self.create_protected_region_meshes()
            self.add_protected_regions_to_plot(protected_region_meshes)
        
        # Add collision markers
        self.add_collision_markers_to_plot()
        
        # Add interactive features
        interactive_text = "Interactive Trajectory Visualization\nUse mouse to rotate, zoom, and pan"
        interactive_text += "\nBlue cylinders show trajectories"
        if self.missiles:
            interactive_text += "\nOrange cylinders show missiles"
        if self.collision_events:
            interactive_text += "\nGreen cylinders show intercepted objects"
        if self.radars:
            interactive_text += "\nPurple spheres show radar stations"
        if self.protected_regions:
            interactive_text += "\nRed cylinders show protected regions"
        if self.collision_events:
            interactive_text += "\nRed X marks show collision points"
        interactive_text += "\nLarge arrows show end positions"
        if satellite_path:
            interactive_text += "\nSatellite imagery overlay enabled"
        
        self.plotter.add_text(
            interactive_text,
            position='upper_left',
            font_size=14,
            color='white'
        )
        
        # Show interactive window
        self.plotter.show()

    def add_collision_markers_to_plot(self):
        """Add collision markers as X symbols to the visualization."""
        if not self.collision_events:
            return
        
        for collision in self.collision_events:
            # Create an X symbol at the collision point
            pos = collision.position
            size = 50  # Size of the X symbol
            
            # Create two perpendicular lines to form an X
            # Line 1: diagonal from top-left to bottom-right
            x1_start = pos[0] - size/2
            y1_start = pos[1] - size/2
            z1_start = pos[2] - size/2
            x1_end = pos[0] + size/2
            y1_end = pos[1] + size/2
            z1_end = pos[2] + size/2
            
            # Line 2: diagonal from top-right to bottom-left
            x2_start = pos[0] + size/2
            y2_start = pos[1] - size/2
            z2_start = pos[2] - size/2
            x2_end = pos[0] - size/2
            y2_end = pos[1] + size/2
            z2_end = pos[2] + size/2
            
            # Create line meshes
            line1 = pv.lines_from_points([[x1_start, y1_start, z1_start], [x1_end, y1_end, z1_end]])
            line2 = pv.lines_from_points([[x2_start, y2_start, z2_start], [x2_end, y2_end, z2_end]])
            
            # Add X lines to plot
            self.plotter.add_mesh(
                line1,
                color='red',
                line_width=8,
                render_lines_as_tubes=True
            )
            self.plotter.add_mesh(
                line2,
                color='red',
                line_width=8,
                render_lines_as_tubes=True
            )

    def add_protected_regions_to_plot(self, protected_region_meshes: Dict[int, pv.PolyData]):
        """Add protected region meshes to the visualization."""
        for region_id, mesh_data in protected_region_meshes.items():
            region = mesh_data['region']
            mesh = mesh_data['mesh']
            
            # Add cylinder mesh with semi-transparent red color
            self.plotter.add_mesh(
                mesh,
                color='red',
                opacity=0.6,  # Increased from 0.3 to 0.6 (30% more opaque)
                show_edges=True,
                edge_color='darkred',
                line_width=2,
                lighting=True,
                ambient=0.4,
                diffuse=0.6,
                specular=0.2
            )
            
            # Add label at the top of the cylinder
            label_pos = (region.centroid[0], region.centroid[1], region.height_limit + 5)
            self.plotter.add_point_labels(
                [label_pos],
                [region.name],
                font_size=14,
                bold=True,
                text_color='red',
                shape_color='red',
                shape_opacity=0.7
            )
            
            # Add radius and height info
            info_text = f"R: {region.radius}m\nH: {region.height_limit}m"
            info_pos = (region.centroid[0], region.centroid[1], region.height_limit / 2)
            self.plotter.add_point_labels(
                [info_pos],
                [info_text],
                font_size=10,
                bold=False,
                text_color='darkred'
            )

    def add_end_position_arrows(self, trajectory_meshes: Dict[int, pv.PolyData], missile_meshes: Dict[int, pv.PolyData] = None):
        """Add arrows at the end positions of trajectories and missiles."""
        # Add end position arrows for trajectories
        for trajectory_id, mesh_data in trajectory_meshes.items():
            if mesh_data['positions'].shape[0] > 0:
                end_position = mesh_data['positions'][-1]
                end_velocity = mesh_data['velocities'][-1]
                
                # Create arrow pointing in the direction of final velocity
                if np.linalg.norm(end_velocity) > 0:
                    vel_norm = end_velocity / np.linalg.norm(end_velocity)
                    arrow = pv.Arrow(
                        start=end_position,
                        direction=vel_norm,
                        scale=40.0,  # Much larger arrow for end position
                        tip_length=0.4,
                        tip_radius=0.5,
                        shaft_radius=0.25
                    )
                    
                    color = 'blue'  # Use blue for all trajectories
                    self.plotter.add_mesh(
                        arrow,
                        color=color,
                        opacity=0.9,
                        lighting=True,
                        ambient=0.8,
                        diffuse=0.2,
                        specular=0.5,
                        specular_power=10
                    )
                    
                    # Add end position label
                    self.plotter.add_point_labels(
                        [end_position],
                        [f'End T{trajectory_id}'],
                        font_size=14,
                        bold=True,
                        text_color=color
                    )
        
        # Add end position arrows for missiles
        if missile_meshes:
            for missile_id, mesh_data in missile_meshes.items():
                if mesh_data['positions'].shape[0] > 0:
                    end_position = mesh_data['positions'][-1]
                    end_velocity = mesh_data['velocities'][-1]
                    
                    # Create arrow pointing in the direction of final velocity
                    if np.linalg.norm(end_velocity) > 0:
                        vel_norm = end_velocity / np.linalg.norm(end_velocity)
                        arrow = pv.Arrow(
                            start=end_position,
                            direction=vel_norm,
                            scale=50.0,  # Much larger arrow for missile end position
                            tip_length=0.4,
                            tip_radius=0.6,
                            shaft_radius=0.3
                        )
                        
                        color = 'yellow'  # All missiles use yellow color
                        self.plotter.add_mesh(
                            arrow,
                            color=color,
                            opacity=0.9,
                            lighting=True,
                            ambient=0.8,
                            diffuse=0.2,
                            specular=0.5,
                            specular_power=10
                        )
                        
                        # Add end position label
                        self.plotter.add_point_labels(
                            [end_position],
                            [f'End M{missile_id}'],
                            font_size=14,
                            bold=True,
                            text_color=color
                        )

    def add_radars_to_plot(self):
        """Add radar stations to the visualization as labeled dots."""
        if not self.radars:
            return
        
        for radar in self.radars:
            # Create a small sphere to represent the radar station
            radar_sphere = pv.Sphere(
                center=radar.emplacement,
                radius=20.0  # Small radius for radar representation
            )
            
            # Add radar sphere to plot
            self.plotter.add_mesh(
                radar_sphere,
                color='purple',  # Purple color for radar stations
                opacity=1.0,  # Increased from 0.8 to 1.0 (30% more opaque)
                lighting=True,
                ambient=0.6,
                diffuse=0.4,
                specular=0.3,
                specular_power=5
            )
            
            # Add radar label
            label_position = (
                radar.emplacement[0],
                radar.emplacement[1],
                radar.emplacement[2] + 50  # Slightly above the radar
            )
            self.plotter.add_point_labels(
                [label_position],
                [f"Radar {radar.id}: {radar.description}"],
                font_size=12,
                bold=True,
                text_color='purple',
                shape_color='purple',
                shape_opacity=0.7
            )

    def _setup_topdown_camera(self, video_zoom: float = 1.0):
        """Setup camera position for top-down view (looking straight down from above)."""
        # Calculate bounds including trajectories, protected regions, and topography
        all_points = []
        
        # Add trajectory points
        for trajectory in self.trajectories:
            for point in trajectory.points:
                all_points.append(point.position)
        
        # Add missile points
        for missile in self.missiles:
            for point in missile.points:
                all_points.append(point.position)
        
        # Add protected region points (centroids and top points)
        for region in self.protected_regions:
            all_points.append(region.centroid)
            # Add top point of cylinder
            top_point = (region.centroid[0], region.centroid[1], region.height_limit)
            all_points.append(top_point)
            # Add points around the cylinder perimeter at different heights
            for angle in np.linspace(0, 2*np.pi, 8):
                x = region.centroid[0] + region.radius * np.cos(angle)
                y = region.centroid[1] + region.radius * np.sin(angle)
                all_points.append((x, y, 0))  # Base
                all_points.append((x, y, region.height_limit))  # Top
        
        if all_points:
            points_array = np.array(all_points)
            center = points_array.mean(axis=0)
            
            # Calculate bounds
            x_min, y_min, z_min = points_array.min(axis=0)
            x_max, y_max, z_max = points_array.max(axis=0)
            
            # Calculate ranges
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            max_range = max(x_range, y_range, z_range)
            
            # Set camera position for top-down view
            camera_distance = max_range * 0.6 / video_zoom  # Apply zoom factor
            
            # Position camera directly above the center, looking straight down
            camera_pos = (
                center[0],  # Same X as center
                center[1],  # Same Y as center  
                center[2] + camera_distance  # Above the scene
            )
            
            self.plotter.camera_position = [
                camera_pos,
                center,  # Look at center
                (0, 1, 0)  # Up vector pointing North (Y-axis)
            ]
            
            # Apply additional zoom factor
            self.plotter.camera.zoom(video_zoom)
            print(f"Top-down camera set with zoom factor: {video_zoom}")
        else:
            # Fallback camera position for top-down
            self.plotter.camera_position = 'xy'  # Top-down preset
            self.plotter.camera.zoom(video_zoom)

    def animate_trajectories_topdown(self, trajectory_meshes: Dict[int, pv.PolyData], 
                                   missile_meshes: Dict[int, pv.PolyData] = None,
                                   fps: int = 10, save_path: str = None, video_zoom: float = 1.0,
                                   intermediate_frames: int = 4):
        """Create a top-down animation of the trajectories and missiles."""
        if not self.animation_data:
            self.create_animation_data(trajectory_meshes, missile_meshes, intermediate_frames)
        
        print(f"Creating top-down animation with {len(self.animation_data)} frames...")
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_paths = []
            
            # Use off_screen plotter for animation
            self.setup_visualization(off_screen=True)
            
            # Add static elements once (topography, protected regions, collision markers)
            print("Adding static elements for top-down view...")
            self.add_topography_to_plot()
            
            # Create protected region meshes for animation
            protected_region_meshes = {}
            if self.protected_regions:
                protected_region_meshes = self.create_protected_region_meshes()
                self.add_protected_regions_to_plot(protected_region_meshes)
            
            # Add collision markers (static throughout animation)
            self.add_collision_markers_to_plot()
            
            # Calculate optimal camera position for top-down view
            self._setup_topdown_camera(video_zoom)
            
            # Track dynamic actors to clear each frame
            dynamic_actors = []
            
            for i, frame_data in enumerate(self.animation_data):
                # Clear only dynamic actors from previous frame
                for actor in dynamic_actors:
                    self.plotter.remove_actor(actor)
                dynamic_actors.clear()
                
                # Add coordinate axes for top-down view (only X and Y axes visible)
                axes_actor = self.plotter.add_axes(
                    xlabel='East (m)', 
                    ylabel='North (m)', 
                    zlabel='',  # Hide Z-axis label for top-down view
                    line_width=2,
                    labels_off=False
                )
                dynamic_actors.append(axes_actor)
                
                # Add time display (dynamic - changes each frame)
                time_actor = self.plotter.add_text(
                    f"Time: {frame_data['time']:.2f}s",
                    position='upper_left',
                    font_size=20,
                    color='white'
                )
                dynamic_actors.append(time_actor)
                
                # Add active trajectories for this frame
                for trajectory_info in frame_data['active_trajectories']:
                    trajectory_id = trajectory_info['id']
                    current_position = trajectory_info['current_position']
                    current_velocity = trajectory_info['current_velocity']
                    trail_positions = trajectory_info['trail_positions']
                    
                    # Check if this trajectory has been intercepted
                    trajectory_intercepted = False
                    for collision in self.collision_events:
                        if trajectory_id in collision.participants and frame_data['time'] >= collision.time:
                            trajectory_intercepted = True
                            break
                    
                    # Choose color based on interception status
                    if trajectory_intercepted:
                        color = 'green'  # Intercepted trajectories are green
                    else:
                        color = 'blue'   # Normal trajectories are blue
                    
                    # Create trail line
                    if len(trail_positions) > 1:
                        trail_points = np.array(trail_positions)
                        trail_line = pv.lines_from_points(trail_points)
                        
                        # Add trail with fading opacity
                        trail_actor = self.plotter.add_mesh(
                            trail_line,
                            color=color,
                            line_width=8,  # Same as missiles
                            opacity=0.8,
                            render_lines_as_tubes=True
                        )
                        dynamic_actors.append(trail_actor)
                    
                    # Add velocity vector (arrow head) for current position
                    if np.linalg.norm(current_velocity) > 0:
                        vel_norm = current_velocity / np.linalg.norm(current_velocity)
                        arrow = pv.Arrow(
                            start=current_position,
                            direction=vel_norm,
                            scale=12.0,  # Same as missiles
                            tip_length=0.4,
                            tip_radius=0.3,
                            shaft_radius=0.15
                        )
                        
                        vector_actor = self.plotter.add_mesh(
                            arrow,
                            color=color,
                            opacity=0.9
                        )
                        dynamic_actors.append(vector_actor)
                
                # Add active missiles for this frame
                for missile_info in frame_data['active_missiles']:
                    missile_id = missile_info['id']
                    current_position = missile_info['current_position']
                    current_velocity = missile_info['current_velocity']
                    trail_positions = missile_info['trail_positions']
                    
                    # All missiles use yellow color
                    color = 'yellow'
                    
                    # Create trail line for missile
                    if len(trail_positions) > 1:
                        trail_points = np.array(trail_positions)
                        trail_line = pv.lines_from_points(trail_points)
                        
                        # Add missile trail with fading opacity
                        trail_actor = self.plotter.add_mesh(
                            trail_line,
                            color=color,
                            line_width=8,
                            opacity=0.8,
                            render_lines_as_tubes=True
                        )
                        dynamic_actors.append(trail_actor)
                    
                    # Add velocity vector (arrow head) for current missile position
                    if np.linalg.norm(current_velocity) > 0:
                        vel_norm = current_velocity / np.linalg.norm(current_velocity)
                        arrow = pv.Arrow(
                            start=current_position,
                            direction=vel_norm,
                            scale=12.0,  # Same as trajectories
                            tip_length=0.4,
                            tip_radius=0.3,
                            shaft_radius=0.15
                        )
                        
                        vector_actor = self.plotter.add_mesh(
                            arrow,
                            color=color,
                            opacity=0.9
                        )
                        dynamic_actors.append(vector_actor)
                
                # Add explosion effects for this frame
                for explosion_info in frame_data['explosions']:
                    position = explosion_info['position']
                    radius = explosion_info['radius']
                    time_factor = explosion_info['time_factor']
                    participants = explosion_info['participants']
                    
                    # Create explosion mesh
                    explosion_mesh = self.create_explosion_mesh(position, radius, time_factor)
                    
                    # Add explosion to plot with dynamic color and opacity
                    explosion_color = 'yellow' if time_factor > 0.5 else 'orange'
                    explosion_opacity = min(0.9, time_factor * 2)  # Fade out over time
                    
                    explosion_actor = self.plotter.add_mesh(
                        explosion_mesh,
                        color=explosion_color,
                        opacity=explosion_opacity,
                        show_edges=False,
                        lighting=True,
                        ambient=0.8,
                        diffuse=0.2
                    )
                    dynamic_actors.append(explosion_actor)
                
                # Add legend for top-down view
                legend_text = f"Top-Down Animation Frame {i+1}/{len(self.animation_data)}\nBlue: Trajectories\nYellow: Missiles"
                if self.collision_events:
                    legend_text += "\nGreen: Intercepted Objects"
                if self.protected_regions:
                    legend_text += "\nRed Cylinders: Protected Regions"
                legend_text += "\nView: Looking Down from Above"
                
                legend_actor = self.plotter.add_text(
                    legend_text,
                    position='upper_right',
                    font_size=16,
                    color='white'
                )
                dynamic_actors.append(legend_actor)
                
                # Save frame
                frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
                self.plotter.screenshot(frame_path, window_size=(1920, 1080))
                frame_paths.append(frame_path)
                
                if (i + 1) % 10 == 0:
                    print(f"  Generated frame {i+1}/{len(self.animation_data)}")
            
            # Create video from frames
            if save_path:
                print(f"Creating top-down video: {save_path}")
                self._create_video_from_frames(frame_paths, save_path, fps)
                print(f"Top-down animation saved to: {save_path}")
            else:
                print("Top-down animation frames generated (no save path specified)")

def main():
    """Main function to run the trajectory visualization suite."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Trajectory Visualization Suite - Choose visualization mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sim_visualize_tracks.py --mode satellite    # Use satellite imagery
  python sim_visualize_tracks.py --mode topography   # Use topographic map
  python sim_visualize_tracks.py --mode auto         # Auto-detect (default)
  python sim_visualize_tracks.py --topdown           # Generate both 3D and top-down videos
  python sim_visualize_tracks.py --topdown --video-zoom 1.5  # Top-down with zoom
        """
    )
    parser.add_argument(
        '--mode', 
        choices=['satellite', 'topography', 'auto'],
        default='auto',
        help='Visualization mode: satellite (satellite imagery), topography (elevation colormap), or auto (auto-detect)'
    )
    parser.add_argument(
        '--photo-zoom',
        type=float,
        default=1.0,
        help='Camera zoom factor for static photos (default: 1.0, higher values = more zoomed in)'
    )
    parser.add_argument(
        '--video-zoom',
        type=float,
        default=1.0,
        help='Camera zoom factor for video animations (default: 1.0, higher values = more zoomed in)'
    )
    parser.add_argument(
        '--intermediate-frames',
        type=int,
        default=4,
        help='Number of intermediate frames between global time steps (default: 4, higher values = smoother animation)'
    )
    parser.add_argument(
        '--static-only',
        action='store_true',
        help='Skip animation generation and only create static visualization'
    )
    parser.add_argument(
        '--topology-offset',
        type=float,
        default=300.0,
        help='Topography offset in meters - adjusts where the terrain surface appears relative to z=0 (default: 300.0, higher values = terrain appears lower)'
    )
    parser.add_argument(
        '--topdown',
        action='store_true',
        help='Generate both regular 3D video and top-down video (looking straight down from above)'
    )
    
    args = parser.parse_args()
    
    visualizer = None
    try:
        print("Trajectory Visualization Suite")
        print("=" * 50)
        print(f"Visualization mode: {args.mode}")
        print("-" * 50)
        
        # Setup WSLg-specific configurations
        configure_pyvista_for_wslg()
        display_ok = setup_wslg_display()
        
        # Load reference location
        print("Loading reference LLA...")
        visualizer = TrajectoryVisualizer(visualization_mode=args.mode, topology_offset=args.topology_offset)
        ref_origin = visualizer.load_reference_lla_from_json('input.json')
        
        # Load global time steps
        print("Loading global time steps...")
        visualizer.load_time_steps_from_json('input.json')
        
        # Load trajectory data
        print("Loading trajectory data...")
        trajectories = visualizer.load_trajectories_from_json('input.json')
        print(f"Loaded {len(trajectories)} trajectories")
        
        # Load protected regions
        print("Loading protected regions...")
        protected_regions = visualizer.load_protected_regions_from_json('input.json')
        print(f"Loaded {len(protected_regions)} protected regions")
        
        # Load missile data
        print("Loading missile data...")
        missiles = visualizer.load_missiles_from_json('input.json')
        print(f"Loaded {len(missiles)} missiles")
        
        # Load radar data
        print("Loading radar data...")
        radars = visualizer.load_radars_from_json('input.json')
        print(f"Loaded {len(radars)} radar stations")
        
        # Detect collisions
        print("Detecting collisions...")
        collisions = visualizer.detect_collisions()
        print(f"Detected {len(collisions)} collision events")
        
        # Determine visualization mode and satellite path
        satellite_path = None
        satellite_data_dir = "satellite_data"
        
        if args.mode == 'satellite':
            # Force satellite mode - check if satellite data exists
            if os.path.exists(satellite_data_dir):
                # First look for quick version for faster processing
                quick_path = os.path.join(satellite_data_dir, "naip_quick.tif")
                if os.path.exists(quick_path):
                    satellite_path = quick_path
                    file_size_mb = os.path.getsize(satellite_path) / (1024 * 1024)
                    print(f"Using quick satellite imagery: {satellite_path} ({file_size_mb:.1f}MB)")
                else:
                    # Fallback to high-resolution files if quick version not available
                    jp2_files = [f for f in os.listdir(satellite_data_dir) if f.lower().endswith('.jp2')]
                    if jp2_files:
                        # Use the largest JP2 file (highest resolution)
                        jp2_files.sort(key=lambda x: os.path.getsize(os.path.join(satellite_data_dir, x)), reverse=True)
                        satellite_path = os.path.join(satellite_data_dir, jp2_files[0])
                        file_size_mb = os.path.getsize(satellite_path) / (1024 * 1024)
                        print(f"Using high-resolution JP2 satellite imagery: {satellite_path} ({file_size_mb:.1f}MB)")
                    else:
                        # Look for other satellite image formats
                        for ext in ['.tif', '.tiff', '.bil', '.img', '.jpg', '.jpeg', '.png', '.jpeg2000']:
                            for file in os.listdir(satellite_data_dir):
                                if file.lower().endswith(ext):
                                    satellite_path = os.path.join(satellite_data_dir, file)
                                    file_size_mb = os.path.getsize(satellite_path) / (1024 * 1024)
                                    print(f"Using satellite imagery: {satellite_path} ({file_size_mb:.1f}MB)")
                                    break
                            if satellite_path:
                                break
                
                if not satellite_path:
                    print("ERROR: Satellite mode requested but no satellite imagery found in satellite_data/ directory")
                    print("Please ensure satellite imagery files are present in the satellite_data/ directory")
                    return
            else:
                print("ERROR: Satellite mode requested but satellite_data/ directory not found")
                print("Please ensure satellite imagery files are present in the satellite_data/ directory")
                return
                
        elif args.mode == 'topography':
            # Force topography mode - ignore satellite data
            print("Using topographic elevation colormap (satellite imagery disabled)")
            satellite_path = None
            
        else:  # args.mode == 'auto'
            # Auto-detect mode (original behavior)
            if os.path.exists(satellite_data_dir):
                # First look for quick version for faster processing
                quick_path = os.path.join(satellite_data_dir, "naip_quick.tif")
                if os.path.exists(quick_path):
                    satellite_path = quick_path
                    file_size_mb = os.path.getsize(satellite_path) / (1024 * 1024)
                    print(f"Auto-detected quick satellite imagery: {satellite_path} ({file_size_mb:.1f}MB)")
                else:
                    # Fallback to high-resolution files if quick version not available
                    jp2_files = [f for f in os.listdir(satellite_data_dir) if f.lower().endswith('.jp2')]
                    if jp2_files:
                        # Use the largest JP2 file (highest resolution)
                        jp2_files.sort(key=lambda x: os.path.getsize(os.path.join(satellite_data_dir, x)), reverse=True)
                        satellite_path = os.path.join(satellite_data_dir, jp2_files[0])
                        file_size_mb = os.path.getsize(satellite_path) / (1024 * 1024)
                        print(f"Auto-detected high-resolution JP2 satellite imagery: {satellite_path} ({file_size_mb:.1f}MB)")
                    else:
                        # Look for other satellite image formats
                        for ext in ['.tif', '.tiff', '.bil', '.img', '.jpg', '.jpeg', '.png', '.jpeg2000']:
                            for file in os.listdir(satellite_data_dir):
                                if file.lower().endswith(ext):
                                    satellite_path = os.path.join(satellite_data_dir, file)
                                    file_size_mb = os.path.getsize(satellite_path) / (1024 * 1024)
                                    print(f"Auto-detected satellite imagery: {satellite_path} ({file_size_mb:.1f}MB)")
                                    break
                            if satellite_path:
                                break
        
        # Create static visualization
        print("Creating static visualization...")
        if satellite_path:
            print(f"Using satellite imagery: {satellite_path}")
            visualizer.create_static_visualization('output/trajectory_static.png', satellite_path, photo_zoom=args.photo_zoom)
        else:
            print("Using elevation colormap for topography")
            visualizer.create_static_visualization('output/trajectory_static.png', photo_zoom=args.photo_zoom)
        
        # Create animation (unless static-only mode is enabled)
        if not args.static_only:
            print("Creating animation...")
            trajectory_meshes = visualizer.create_trajectory_meshes()
            missile_meshes = visualizer.create_missile_meshes()
            visualizer.animate_trajectories(trajectory_meshes, missile_meshes, fps=5, save_path='output/trajectory_animation.mp4', video_zoom=args.video_zoom, intermediate_frames=args.intermediate_frames)
            
            # Create top-down animation if requested
            if args.topdown:
                print("Creating top-down animation...")
                visualizer.animate_trajectories_topdown(trajectory_meshes, missile_meshes, fps=5, save_path='output/trajectory_animation_topdown.mp4', video_zoom=args.video_zoom, intermediate_frames=args.intermediate_frames)
        else:
            print("Skipping animation generation (static-only mode)")
        
        # Create interactive visualization with WSLg optimization
        print("Creating interactive visualization...")
        if not display_ok:
            print("Display issues detected. Interactive visualization may not work properly.")
            print("   Consider using one of these alternatives:")
            print("   1. Run on Windows directly (not WSL)")
            print("   2. Update WSL for WSLg support: wsl --update")
            print("   3. Use the static visualization and animation files instead")
            
            # Ask user if they want to continue
            try:
                response = input("Continue with interactive visualization anyway? (y/N): ").strip().lower()
                if response != 'y':
                    print("Skipping interactive visualization. Check the generated files:")
                    print("   - output/trajectory_static.png (static visualization)")
                    print("   - output/trajectory_animation.mp4 (animation)")
                    if args.topdown and not args.static_only:
                        print("   - output/trajectory_animation_topdown.mp4 (top-down animation)")
                    return
            except KeyboardInterrupt:
                print("\nSkipping interactive visualization.")
                return
        
        # Check if running in WSLg for better messaging
        is_wslg = False
        try:
            with open('/proc/version', 'r') as f:
                content = f.read().lower()
                if 'microsoft' in content and 'wslg' in content:
                    is_wslg = True
        except:
            pass
        
        if is_wslg:
            print("WSLg detected - interactive visualization should work well!")
        
        try:
            if satellite_path:
                visualizer.create_interactive_visualization(satellite_path)
            else:
                visualizer.create_interactive_visualization()
        except Exception as e:
            print(f"Interactive visualization failed: {e}")
            if is_wslg:
                print("This is unexpected with WSLg. Try updating WSL: wsl --update")
            else:
                print("This is common in WSL. The static visualization and animation files should still work.")
            print("Generated files:")
            print("   - output/trajectory_static.png (static visualization)")
            print("   - output/trajectory_animation.mp4 (animation)")
            if args.topdown and not args.static_only:
                print("   - output/trajectory_animation_topdown.mp4 (top-down animation)")
        
        print("Visualization complete!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        if visualizer is not None:
            try:
                visualizer.close()
            except:
                pass
        print("Cleanup complete")

if __name__ == "__main__":
    main()