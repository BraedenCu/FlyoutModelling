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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

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
        self._setup_transformers()
        
    def _setup_transformers(self):
        lat0, lon0, h0 = self.ref_origin
        # WGS84 geodetic
        self.transformer_enu2llh = Transformer.from_crs(
            f"epsg:4978",
            f"epsg:4326",
            always_xy=True
        )
        # ENU local tangent plane
        self.transformer_llh2enu = Transformer.from_crs(
            f"epsg:4326",
            f"epsg:4978",
            always_xy=True
        )
    
    def enu_to_latlon(self, x, y, z):
        # Use pyproj for ENU to ECEF to LLH conversion
        # For small areas, use a simple local tangent plane approximation
        lat0, lon0, h0 = self.ref_origin
        # Use pyproj's built-in ENU projection
        proj_enu = Proj(proj='aeqd', lat_0=lat0, lon_0=lon0, ellps='WGS84')
        lon, lat = proj_enu(x, y, inverse=True)
        return lat, lon
    
    def get_dem_data(self, bounds: Tuple[float, float, float, float], 
                     resolution: int = 30) -> Optional[np.ndarray]:
        """
        Download DEM data from OpenTopography for the specified bounds.
        Args:
            bounds: (x_min, x_max, y_min, y_max) in ENU meters
            resolution: DEM resolution in meters (30, 10, 3, 1)
        Returns:
            (X, Y, Z) meshgrid in ENU meters, or None if failed
        """
        try:
            # Convert ENU bounds to lat/lon using reference origin
            x_min, x_max, y_min, y_max = bounds
            lat0, lon0, h0 = self.ref_origin
            # Four corners
            lat_sw, lon_sw = self.enu_to_latlon(x_min, y_min, 0)
            lat_ne, lon_ne = self.enu_to_latlon(x_max, y_max, 0)
            south, north = min(lat_sw, lat_ne), max(lat_sw, lat_ne)
            west, east = min(lon_sw, lon_ne), max(lon_sw, lon_ne)
            
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
                xs = np.arange(width)
                ys = np.arange(height)
                X, Y = np.meshgrid(xs, ys)
                X, Y = rasterio.transform.xy(transform, Y, X, offset="center")
                X = np.array(X)
                Y = np.array(Y)
                # Convert lat/lon grid to ENU meters for visualization
                proj_enu = Proj(proj='aeqd', lat_0=lat0, lon_0=lon0, ellps='WGS84')
                X_enu, Y_enu = proj_enu(X, Y)
                # Ensure arrays are 2D
                X_enu = np.array(X_enu).reshape(Z.shape)
                Y_enu = np.array(Y_enu).reshape(Z.shape)
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
    
    def calculate_bounds(self) -> Tuple[float, float, float, float]:
        """Calculate the bounding box for all trajectories."""
        if not self.trajectories:
            return (-100, 100, -100, 100)
        
        all_positions = []
        for trajectory in self.trajectories:
            for point in trajectory.points:
                all_positions.append(point.position)
        
        positions = np.array(all_positions)
        x_min, y_min = positions[:, 0].min(), positions[:, 1].min()
        x_max, y_max = positions[:, 0].max(), positions[:, 1].max()
        
        # Add padding to create a reasonable bounding box for DEM data
        padding = 500  # 500 meters padding for real-world DEM data
        return (x_min - padding, x_max + padding, y_min - padding, y_max + padding)
    
    def setup_topography(self, bounds: Tuple[float, float, float, float]):
        """Setup topographic data for visualization."""
        print("Setting up topography...")
        
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
        
        self.topography_mesh = grid
        
        # Create a surface mesh for better visualization
        surface = grid.extract_surface()
        surface = surface.smooth(n_iter=10, relaxation_factor=0.1)
        
        # Add elevation data as scalar field
        surface.point_data['elevation'] = Z.flatten()
        
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
    
    def setup_visualization(self, window_size: Tuple[int, int] = (1920, 1080), off_screen: bool = False):
        """Setup the main visualization window."""
        # Create plotter
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
        """Add topography to the visualization."""
        if self.topography_mesh is None:
            return
        
        # Create custom colormap for topography
        colors = ['darkgreen', 'forestgreen', 'yellow', 'orange', 'brown', 'white']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('terrain', colors, N=n_bins)
        
        # Add topography mesh
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
            
            # Add trajectory line as tube for better visibility
            tube = mesh_data['line'].tube(radius=0.5)
            self.plotter.add_mesh(
                tube,
                color=color,
                line_width=4,
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
    
    def create_animation_data(self, trajectory_meshes: Dict[int, pv.PolyData]):
        """Prepare data for animation."""
        self.animation_data = []
        
        # Get all unique timestamps
        all_times = set()
        for mesh_data in trajectory_meshes.values():
            all_times.update(mesh_data['times'])
        
        sorted_times = sorted(all_times)
        
        for time in sorted_times:
            frame_data = {'time': time, 'active_trajectories': []}
            
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
            
            self.animation_data.append(frame_data)
    
    def animate_trajectories(self, trajectory_meshes: Dict[int, pv.PolyData], 
                           fps: int = 10, save_path: str = None):
        """Create an animation of the trajectories."""
        if not self.animation_data:
            self.create_animation_data(trajectory_meshes)
        
        print(f"Creating animation with {len(self.animation_data)} frames...")
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_paths = []
            
            # Use off_screen plotter for animation
            self.setup_visualization(off_screen=True)
            self.add_topography_to_plot()
            
            for i, frame_data in enumerate(self.animation_data):
                # Clear previous trajectory meshes
                self.plotter.clear_actors()
                
                # Re-add topography
                self.add_topography_to_plot()
                
                # Add coordinate axes
                self.plotter.add_axes(
                    xlabel='East (m)', 
                    ylabel='North (m)', 
                    zlabel='Up (m)',
                    line_width=2,
                    labels_off=False
                )
                
                # Add time display
                self.plotter.add_text(
                    f"Time: {frame_data['time']:.2f}s",
                    position='upper_left',
                    font_size=20,
                    color='white'
                )
                
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
                    
                    self.plotter.add_mesh(
                        points,
                        color=color,
                        point_size=15,
                        render_points_as_spheres=True,
                        show_scalar_bar=False
                    )
                    
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
                            
                            self.plotter.add_mesh(
                                combined_vectors,
                                color=color,
                                opacity=0.8
                            )
                
                # Save frame
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                self.plotter.screenshot(frame_path, window_size=(1920, 1080))
                frame_paths.append(frame_path)
            
            # Create video from frames
            if save_path:
                self._create_video_from_frames(frame_paths, save_path, fps)
                print(f"Animation saved to: {save_path}")
    
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
    
    def create_static_visualization(self, save_path: str = None):
        """Create a static visualization of all trajectories."""
        if not self.trajectories:
            print("No trajectories loaded!")
            return
        
        # Calculate bounds
        bounds = self.calculate_bounds()
        
        # Setup topography
        self.setup_topography(bounds)
        
        # Setup visualization (off_screen if saving)
        self.setup_visualization(off_screen=bool(save_path))
        
        # Add topography
        self.add_topography_to_plot()
        
        # Create trajectory meshes
        trajectory_meshes = self.create_trajectory_meshes()
        
        # Add trajectories
        self.add_trajectories_to_plot(trajectory_meshes)
        
        # Add legend
        self.plotter.add_text(
            "Trajectory Visualization\nRed: Trajectory 1\nBlue: Trajectory 2",
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
    
    def create_interactive_visualization(self):
        """Create an interactive visualization."""
        if not self.trajectories:
            print("No trajectories loaded!")
            return
        
        # Calculate bounds
        bounds = self.calculate_bounds()
        
        # Setup topography
        self.setup_topography(bounds)
        
        # Setup visualization
        self.setup_visualization()
        
        # Add topography
        self.add_topography_to_plot()
        
        # Create trajectory meshes
        trajectory_meshes = self.create_trajectory_meshes()
        
        # Add trajectories
        self.add_trajectories_to_plot(trajectory_meshes)
        
        # Add interactive features
        self.plotter.add_text(
            "Interactive Trajectory Visualization\nUse mouse to rotate, zoom, and pan",
            position='upper_left',
            font_size=14,
            color='white'
        )
        
        # Show interactive window
        self.plotter.show()

def main():
    """Main function to demonstrate the visualization suite."""
    print("🚀 Trajectory Visualization Suite")
    print("=" * 50)
    
    # Create visualizer
    visualizer = TrajectoryVisualizer()
    
    # Load trajectories
    print("Loading trajectory data...")
    trajectories = visualizer.load_trajectories_from_json('input.py')
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Create static visualization
    print("Creating static visualization...")
    visualizer.create_static_visualization('trajectory_static.png')
    
    # Create animation
    print("Creating animation...")
    trajectory_meshes = visualizer.create_trajectory_meshes()
    visualizer.animate_trajectories(trajectory_meshes, fps=5, save_path='trajectory_animation.mp4')
    
    # Create interactive visualization
    print("Creating interactive visualization...")
    visualizer.create_interactive_visualization()
    
    print("✅ Visualization complete!")

if __name__ == "__main__":
    main()
