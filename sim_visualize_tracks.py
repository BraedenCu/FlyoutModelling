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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# WSL-specific configuration
def setup_wsl_display():
    """Configure display settings for WSL compatibility."""
    # Check if running in WSL
    is_wsl = False
    try:
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower():
                is_wsl = True
    except:
        pass
    
    if is_wsl:
        print("🔧 WSL detected - configuring display settings...")
        
        # Set environment variables for WSL graphics
        os.environ['DISPLAY'] = ':0'
        os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
        
        # Try to start X server if not running
        try:
            # Check if X server is running
            result = subprocess.run(['xset', 'q'], capture_output=True, timeout=5)
            if result.returncode != 0:
                print("⚠️ X server not detected. Please ensure you have:")
                print("   1. VcXsrv, Xming, or similar X server running on Windows")
                print("   2. DISPLAY=:0 set in your WSL environment")
                print("   3. WSLg enabled (Windows 11) or X server configured (Windows 10)")
                return False
            else:
                print("✅ X server detected and running")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️ Could not verify X server status")
            return False
    
    return True

def configure_pyvista_for_wsl():
    """Configure PyVista settings for WSL compatibility."""
    # Check if running in WSL
    is_wsl = False
    try:
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower():
                is_wsl = True
    except:
        pass
    
    if is_wsl:
        print("🔧 Configuring PyVista for WSL...")
        
        # Set PyVista to use software rendering if needed
        pv.global_theme.renderer = 'opengl2'
        
        # Configure for better WSL compatibility
        pv.global_theme.window_size = [1024, 768]  # Smaller default window
        pv.global_theme.anti_aliasing = 'fxaa'  # Use FXAA instead of MSAA
        pv.global_theme.multi_samples = 1  # Reduce multisampling
        
        # Set fallback options
        pv.global_theme.use_panel = False  # Disable panel for WSL
        pv.global_theme.show_edges = False  # Reduce rendering complexity
        
        print("✅ PyVista configured for WSL")

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
            
            print(f"✅ Successfully created satellite texture: {sat_img.shape}")
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
            print("✅ Successfully loaded local topography data")
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
        """Generate realistic synthetic topography with multiple features."""
        Z = np.zeros_like(X)
        # Base elevation
        Z += 100 + 50 * np.sin(X / 100) * np.cos(Y / 100)
        # Add mountains
        mountain_centers = [(0, 0), (50, 50), (-30, 40)]
        for cx, cy in mountain_centers:
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            Z += 200 * np.exp(-dist**2 / (2 * 30**2))
        # Add valleys
        valley_centers = [(20, -20), (-40, -30)]
        for cx, cy in valley_centers:
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            Z -= 50 * np.exp(-dist**2 / (2 * 40**2))
        # Add noise for realism
        Z += np.random.normal(0, 2, X.shape)
        return Z

class TrajectoryVisualizer:
    """Main visualization class for trajectory display."""
    
    def __init__(self, topography_manager: TopographyManager = None, ref_origin=(0.0, 0.0, 0.0)):
        self.ref_origin = ref_origin
        self.topography_manager = topography_manager or TopographyManager(ref_origin=ref_origin)
        self.plotter = None
        self.trajectories = []
        self.missiles = []
        self.protected_regions = []
        self.collision_events = []
        self.topography_mesh = None
        self.animation_data = []
        
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
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        missiles = []
        if 'missile_history' in data:
            for missile_data in data['missile_history']:
                points = []
                for point_data in missile_data['timesteps']:
                    point = TrajectoryPoint(
                        time=point_data['time'],
                        position=tuple(point_data['position']),
                        velocity=tuple(point_data['velocity'])
                    )
                    points.append(point)
                
                missile = Missile(id=missile_data['id'], points=points)
                missiles.append(missile)
        
        self.missiles = missiles
        return missiles
    
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
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if 'reference_lla' in data:
            ref_data = data['reference_lla']
            ref_lla = (
                ref_data['latitude'],
                ref_data['longitude'], 
                ref_data['altitude']
            )
            self.ref_origin = ref_lla
            # Update topography manager with new reference
            if self.topography_manager:
                self.topography_manager.ref_origin = ref_lla
                self.topography_manager._setup_transformers()
            
            print(f"Loaded reference LLA: {ref_lla}")
            if 'description' in ref_data:
                print(f"Reference location: {ref_data['description']}")
            
            return ref_lla
        else:
            print("No reference_lla found in JSON, using default (0, 0, 0)")
            return self.ref_origin
    
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
    
    def setup_topography(self, bounds: Tuple[float, float, float, float], satellite_path: str = None):
        """Setup topographic data for visualization with optional satellite overlay."""
        print("Setting up topography...")
        
        # Load satellite imagery if provided
        if satellite_path and os.path.exists(satellite_path):
            print("Loading satellite imagery overlay...")
            success = self.topography_manager.satellite_manager.load_satellite_imagery(satellite_path)
            if success:
                print("✅ Satellite imagery loaded successfully")
            else:
                print("⚠️ Failed to load satellite imagery, using topography only")
        
        # Get topographic data
        topo_data = self.topography_manager.get_dem_data(bounds)
        if topo_data is None:
            print("Failed to get topography, using flat surface")
            return
        
        X, Y, Z = topo_data
        
        # Ensure arrays are 2D and have the same shape
        if X.ndim != 2 or Y.ndim != 2 or Z.ndim != 2:
            print("Error: Topography arrays must be 2D")
            return
        
        if X.shape != Y.shape or Y.shape != Z.shape:
            print("Error: Topography arrays must have the same shape")
            return
        
        print(f"Creating topography mesh with shape: {X.shape}")
        
        # Create PyVista mesh
        grid = pv.StructuredGrid()
        grid.points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        grid.dimensions = [X.shape[1], X.shape[0], 1]
        
        # Add elevation data as scalar field
        grid.point_data['elevation'] = Z.flatten()
        
        # Add satellite texture if available
        tex_coords = None
        if self.topography_manager.satellite_manager.satellite_texture is not None:
            print("Adding satellite texture to topography mesh...")
            # Calculate texture coordinates
            tex_coords = self.topography_manager.satellite_manager.get_texture_coordinates(X, Y)
            print(f"Texture coordinates shape: {tex_coords.shape}")
        
        self.topography_mesh = grid
        
        # Create a surface mesh for better visualization
        surface = grid.extract_surface()
        surface = surface.smooth(n_iter=10, relaxation_factor=0.1)
        
        # Add elevation data as scalar field
        surface.point_data['elevation'] = Z.flatten()
        
        # Add texture coordinates to surface if available
        if tex_coords is not None:
            # Use the newer PyVista API
            surface.active_texture_coordinates = tex_coords.astype(np.float32)
            print(f"✅ Set texture coordinates on surface mesh")
            print(f"Texture coordinates shape: {tex_coords.shape}")
            print(f"Texture coordinates range: {tex_coords.min()} to {tex_coords.max()}")
        
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
        # Check if running in WSL and adjust window size accordingly
        is_wsl = False
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    is_wsl = True
        except:
            pass
        
        if is_wsl and not off_screen:
            # Use smaller window size for WSL to avoid display issues
            window_size = (1024, 768)
            print("🔧 Using WSL-optimized window size: 1024x768")
        
        # Create plotter with WSL-friendly settings
        if is_wsl:
            # Use software rendering for better WSL compatibility
            self.plotter = pv.Plotter(
                off_screen=off_screen, 
                window_size=window_size,
                lighting='three lights',  # Use simpler lighting
                multi_samples=1  # Reduce multisampling for WSL
            )
        else:
            self.plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
        
        # Set background
        self.plotter.set_background('black')
        
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
        
        # Check if satellite texture is available
        if self.topography_manager.satellite_manager.satellite_texture is not None:
            print("Adding topography with satellite imagery overlay...")
            # Add topography mesh with satellite texture
            self.plotter.add_mesh(
                self.topography_mesh,
                texture=self.topography_manager.satellite_manager.satellite_texture,
                show_edges=False,
                lighting=True,
                ambient=0.3,
                diffuse=0.7,
                specular=0.2,
                specular_power=15,
                smooth_shading=True,
                opacity=0.95
            )
        else:
            print("Adding topography with elevation colormap...")
            # Create custom colormap for topography
            colors = ['darkgreen', 'forestgreen', 'yellow', 'orange', 'brown', 'white']
            n_bins = 256
            cmap = LinearSegmentedColormap.from_list('terrain', colors, N=n_bins)
            
            # Add topography mesh with elevation colormap
            self.plotter.add_mesh(
                self.topography_mesh,
                scalars='elevation',
                cmap=cmap,
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
        # Color palette for different trajectories
        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
        
        for i, (trajectory_id, mesh_data) in enumerate(trajectory_meshes.items()):
            color = colors[i % len(colors)]
            
            # Add trajectory line as thick tube for better visibility
            tube = mesh_data['line'].tube(radius=3.0)  # Increased from 0.5 to 3.0
            self.plotter.add_mesh(
                tube,
                color=color,
                line_width=10,  # Increased from 4 to 10
                show_scalar_bar=False
            )
            
            # Add velocity vectors
            if mesh_data['positions'].shape[0] > 0:
                self.plotter.add_mesh(
                    mesh_data['velocity_vectors'],
                    color=color,
                    opacity=0.8
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
        # Color palette for missiles (different from trajectories)
        missile_colors = ['orange', 'purple', 'brown', 'pink', 'cyan', 'lime', 'gold', 'coral']
        
        for i, (missile_id, mesh_data) in enumerate(missile_meshes.items()):
            color = missile_colors[i % len(missile_colors)]
            
            # Add missile line as thick tube for better visibility
            tube = mesh_data['line'].tube(radius=15.0)  # Increased from 8.0 to 15.0
            self.plotter.add_mesh(
                tube,
                color=color,
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
                self.plotter.add_mesh(
                    mesh_data['velocity_vectors'],
                    color=color,
                    opacity=0.8
                )
            
            # Add missile ID label
            if mesh_data['positions'].shape[0] > 0:
                start_pos = mesh_data['positions'][0]
                self.plotter.add_point_labels(
                    [start_pos],
                    [f'Missile {missile_id}'],
                    font_size=16,
                    bold=True,
                    text_color=color
                )
    
    def create_animation_data(self, trajectory_meshes: Dict[int, pv.PolyData], 
                            missile_meshes: Dict[int, pv.PolyData] = None):
        """Prepare data for animation."""
        self.animation_data = []
        
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
        
        for time in sorted_times:
            frame_data = {
                'time': time, 
                'active_trajectories': [], 
                'active_missiles': [],
                'explosions': []
            }
            
            # Add trajectory data for this time
            for trajectory_id, mesh_data in trajectory_meshes.items():
                # Find points at this time
                time_indices = np.where(np.isclose(mesh_data['times'], time))[0]
                
                if len(time_indices) > 0:
                    positions = mesh_data['positions'][time_indices]
                    velocities = mesh_data['velocities'][time_indices]
                    
                    frame_data['active_trajectories'].append({
                        'id': trajectory_id,
                        'positions': positions,
                        'velocities': velocities
                    })
            
            # Add missile data for this time
            if missile_meshes:
                for missile_id, mesh_data in missile_meshes.items():
                    # Find points at this time
                    time_indices = np.where(np.isclose(mesh_data['times'], time))[0]
                    
                    if len(time_indices) > 0:
                        positions = mesh_data['positions'][time_indices]
                        velocities = mesh_data['velocities'][time_indices]
                        
                        frame_data['active_missiles'].append({
                            'id': missile_id,
                            'positions': positions,
                            'velocities': velocities
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
    
    def animate_trajectories(self, trajectory_meshes: Dict[int, pv.PolyData], 
                           missile_meshes: Dict[int, pv.PolyData] = None,
                           fps: int = 10, save_path: str = None):
        """Create an animation of the trajectories and missiles."""
        if not self.animation_data:
            self.create_animation_data(trajectory_meshes, missile_meshes)
        
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
            self._setup_animation_camera()
            
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
                    positions = trajectory_info['positions']
                    velocities = trajectory_info['velocities']
                    
                    # Create point cloud for current positions
                    points = pv.PolyData(positions)
                    points.point_data['velocity_magnitude'] = np.linalg.norm(velocities, axis=1)
                    
                    # Add trajectory points
                    color = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple'][trajectory_id % 8]
                    
                    trajectory_actor = self.plotter.add_mesh(
                        points,
                        color=color,
                        point_size=15,
                        render_points_as_spheres=True,
                        show_scalar_bar=False
                    )
                    dynamic_actors.append(trajectory_actor)
                    
                    # Add velocity vectors
                    if len(positions) > 0:
                        velocity_vectors = []
                        for pos, vel in zip(positions, velocities):
                            if np.linalg.norm(vel) > 0:
                                vel_norm = vel / np.linalg.norm(vel)
                                arrow = pv.Arrow(
                                    start=pos,
                                    direction=vel_norm,
                                    scale=3.0,
                                    tip_length=0.3,
                                    tip_radius=0.1,
                                    shaft_radius=0.05
                                )
                                velocity_vectors.append(arrow)
                        
                        if velocity_vectors:
                            combined_vectors = velocity_vectors[0]
                            for vec in velocity_vectors[1:]:
                                combined_vectors = combined_vectors.merge(vec)
                            
                            vector_actor = self.plotter.add_mesh(
                                combined_vectors,
                                color=color,
                                opacity=0.8
                            )
                            dynamic_actors.append(vector_actor)
                
                # Add active missiles for this frame
                for missile_info in frame_data['active_missiles']:
                    missile_id = missile_info['id']
                    positions = missile_info['positions']
                    velocities = missile_info['velocities']
                    
                    # Create point cloud for current positions
                    points = pv.PolyData(positions)
                    points.point_data['velocity_magnitude'] = np.linalg.norm(velocities, axis=1)
                    
                    # Add missile points
                    color = ['orange', 'purple', 'brown', 'pink', 'cyan', 'lime', 'gold', 'coral'][missile_id % 8]
                    
                    missile_actor = self.plotter.add_mesh(
                        points,
                        color=color,
                        point_size=40,  # Increased from 25 to 40
                        render_points_as_spheres=True,
                        show_scalar_bar=False
                    )
                    dynamic_actors.append(missile_actor)
                    
                    # Add velocity vectors
                    if len(positions) > 0:
                        velocity_vectors = []
                        for pos, vel in zip(positions, velocities):
                            if np.linalg.norm(vel) > 0:
                                vel_norm = vel / np.linalg.norm(vel)
                                arrow = pv.Arrow(
                                    start=pos,
                                    direction=vel_norm,
                                    scale=3.0,
                                    tip_length=0.3,
                                    tip_radius=0.1,
                                    shaft_radius=0.05
                                )
                                velocity_vectors.append(arrow)
                        
                        if velocity_vectors:
                            combined_vectors = velocity_vectors[0]
                            for vec in velocity_vectors[1:]:
                                combined_vectors = combined_vectors.merge(vec)
                            
                            vector_actor = self.plotter.add_mesh(
                                combined_vectors,
                                color=color,
                                opacity=0.8
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
                legend_text = f"Animation Frame {i+1}/{len(self.animation_data)}\nRed: Trajectory 1\nBlue: Trajectory 2"
                if self.missiles:
                    legend_text += "\nOrange/Purple: Missiles"
                if self.protected_regions:
                    legend_text += "\nRed Cylinders: Protected Regions"
                if self.collision_events:
                    legend_text += "\nRed X: Collision Points"
                
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
    
    def _setup_animation_camera(self):
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
            
            # Set camera position with extra margin to ensure everything is visible
            camera_distance = max_range * 2.0  # Increased multiplier for better framing
            
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
            
            # Zoom out slightly to ensure everything fits
            self.plotter.camera.zoom(0.7)
        else:
            # Fallback camera position
            self.plotter.camera_position = 'iso'
            self.plotter.camera.zoom(1.0)
    
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
    
    def create_static_visualization(self, save_path: str = None, satellite_path: str = None):
        """Create a static visualization of all trajectories with optional satellite overlay."""
        if not self.trajectories:
            print("No trajectories loaded!")
            return
        
        # Calculate bounds
        bounds = self.calculate_bounds()
        
        # Setup topography with optional satellite overlay
        self.setup_topography(bounds, satellite_path)
        
        # Setup visualization (off_screen if saving)
        self.setup_visualization(off_screen=bool(save_path))
        
        # Add topography
        self.add_topography_to_plot()
        
        # Create trajectory meshes
        trajectory_meshes = self.create_trajectory_meshes()
        
        # Add trajectories
        self.add_trajectories_to_plot(trajectory_meshes)
        
        # Create and add missile meshes
        if self.missiles:
            missile_meshes = self.create_missile_meshes()
            self.add_missiles_to_plot(missile_meshes)
        
        # Create and add protected region meshes
        if self.protected_regions:
            protected_region_meshes = self.create_protected_region_meshes()
            self.add_protected_regions_to_plot(protected_region_meshes)
        
        # Add collision markers
        self.add_collision_markers_to_plot()
        
        # Add legend
        legend_text = "Trajectory Visualization\nRed: Trajectory 1\nBlue: Trajectory 2"
        if self.missiles:
            legend_text += "\nOrange/Purple: Missiles"
        if self.protected_regions:
            legend_text += "\nRed Cylinders: Protected Regions"
        if self.collision_events:
            legend_text += "\nRed X: Collision Points"
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
        self.setup_topography(bounds, satellite_path)
        
        # Setup visualization
        self.setup_visualization()
        
        # Add topography
        self.add_topography_to_plot()
        
        # Create trajectory meshes
        trajectory_meshes = self.create_trajectory_meshes()
        
        # Add trajectories
        self.add_trajectories_to_plot(trajectory_meshes)
        
        # Create and add missile meshes
        if self.missiles:
            missile_meshes = self.create_missile_meshes()
            self.add_missiles_to_plot(missile_meshes)
        
        # Create and add protected region meshes
        if self.protected_regions:
            protected_region_meshes = self.create_protected_region_meshes()
            self.add_protected_regions_to_plot(protected_region_meshes)
        
        # Add collision markers
        self.add_collision_markers_to_plot()
        
        # Add interactive features
        interactive_text = "Interactive Trajectory Visualization\nUse mouse to rotate, zoom, and pan"
        if self.missiles:
            interactive_text += "\nOrange/Purple cylinders show missiles"
        if self.protected_regions:
            interactive_text += "\nRed cylinders show protected regions"
        if self.collision_events:
            interactive_text += "\nRed X marks show collision points"
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
                opacity=0.3,
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

def main():
    """Main function to run the trajectory visualization suite."""
    print("🚀 Trajectory Visualization Suite")
    print("=" * 50)
    
    # Setup WSL-specific configurations
    configure_pyvista_for_wsl()
    display_ok = setup_wsl_display()
    
    # Load reference location
    print("Loading reference LLA...")
    visualizer = TrajectoryVisualizer()
    ref_origin = visualizer.load_reference_lla_from_json('input.py')
    
    # Load trajectory data
    print("Loading trajectory data...")
    trajectories = visualizer.load_trajectories_from_json('input.py')
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Load protected regions
    print("Loading protected regions...")
    protected_regions = visualizer.load_protected_regions_from_json('input.py')
    print(f"Loaded {len(protected_regions)} protected regions")
    
    # Load missile data
    print("Loading missile data...")
    missiles = visualizer.load_missiles_from_json('input.py')
    print(f"Loaded {len(missiles)} missiles")
    
    # Detect collisions
    print("Detecting collisions...")
    collisions = visualizer.detect_collisions()
    print(f"Detected {len(collisions)} collision events")
    
    # Check for satellite imagery
    satellite_path = None
    satellite_data_dir = "satellite_data"
    if os.path.exists(satellite_data_dir):
        # First look for quick version, then high-resolution version
        quick_path = os.path.join(satellite_data_dir, "naip_quick.tif")
        if os.path.exists(quick_path):
            satellite_path = quick_path
            print(f"Found quick satellite imagery: {satellite_path}")
        else:
            # Look for high-resolution satellite image formats
            for ext in ['.tif', '.tiff', '.bil', '.img', '.jpg', '.jpeg', '.png', '.jp2', '.jpeg2000']:
                for file in os.listdir(satellite_data_dir):
                    if file.lower().endswith(ext):
                        satellite_path = os.path.join(satellite_data_dir, file)
                        print(f"Found high-resolution satellite imagery: {satellite_path}")
                        break
                if satellite_path:
                    break
    
    # Create static visualization
    print("Creating static visualization...")
    if satellite_path:
        print(f"Using satellite imagery: {satellite_path}")
        visualizer.create_static_visualization('trajectory_static.png', satellite_path)
    else:
        print("No satellite imagery found, using elevation colormap")
        visualizer.create_static_visualization('trajectory_static.png')
    
    # Create animation
    print("Creating animation...")
    trajectory_meshes = visualizer.create_trajectory_meshes()
    missile_meshes = visualizer.create_missile_meshes()
    visualizer.animate_trajectories(trajectory_meshes, missile_meshes, fps=5, save_path='trajectory_animation.mp4')
    
    # Create interactive visualization with WSL fallback
    print("Creating interactive visualization...")
    if not display_ok:
        print("⚠️ Display issues detected. Interactive visualization may not work properly.")
        print("   Consider using one of these alternatives:")
        print("   1. Run on Windows directly (not WSL)")
        print("   2. Use WSLg (Windows 11)")
        print("   3. Install and configure VcXsrv or Xming on Windows")
        print("   4. Use the static visualization and animation files instead")
        
        # Ask user if they want to continue
        try:
            response = input("Continue with interactive visualization anyway? (y/N): ").strip().lower()
            if response != 'y':
                print("Skipping interactive visualization. Check the generated files:")
                print("   - trajectory_static.png (static visualization)")
                print("   - trajectory_animation.mp4 (animation)")
                return
        except KeyboardInterrupt:
            print("\nSkipping interactive visualization.")
            return
    
    try:
        if satellite_path:
            visualizer.create_interactive_visualization(satellite_path)
        else:
            visualizer.create_interactive_visualization()
    except Exception as e:
        print(f"❌ Interactive visualization failed: {e}")
        print("This is common in WSL. The static visualization and animation files should still work.")
        print("Generated files:")
        print("   - trajectory_static.png (static visualization)")
        print("   - trajectory_animation.mp4 (animation)")
    
    print("✅ Visualization complete!")

if __name__ == "__main__":
    main()
