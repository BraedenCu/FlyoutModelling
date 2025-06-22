# Trajectory Visualization Suite

A complete end-to-end visualization system for displaying trajectory tracks with high-fidelity topographic data using PyVista and OpenTopography integration.

## Features

- **3D Trajectory Visualization**: Display multiple trajectories with position and velocity data
- **High-Fidelity Topography**: Synthetic terrain generation with realistic features (mountains, valleys, etc.)
- **Interactive 3D Viewing**: Rotate, zoom, and pan through the visualization
- **Animation Support**: Create time-based animations of trajectory movement
- **Multiple Output Formats**: Static images, interactive windows, and video animations
- **ENU Coordinate System**: Support for East-North-Up coordinate system
- **Velocity Vector Display**: Show velocity vectors at each trajectory point

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. For video creation (optional), install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Usage

### Basic Usage

Run the visualization suite with the provided sample data:

```bash
python sim_visualize_tracks.py --mode topography --photo-zoom 1.5 --video-zoom 0.5
```

This will:
1. Load trajectory data from `input.py`
2. Create a static visualization saved as `trajectory_static.png`
3. Generate an animation saved as `trajectory_animation.mp4`
4. Open an interactive 3D visualization window

### Visualization Modes

The visualization suite supports three different modes for the terrain display:

#### Auto-detect Mode (Default)
```bash
python sim_visualize_tracks.py --mode auto
```
Automatically detects and uses satellite imagery if available in the `satellite_data/` directory, otherwise falls back to topographic elevation colormap.

#### Satellite Imagery Mode
```bash
python sim_visualize_tracks.py --mode satellite
```
Forces the use of satellite imagery. Requires satellite image files to be present in the `satellite_data/` directory. If no satellite data is found, the program will exit with an error.

#### Topography Mode
```bash
python sim_visualize_tracks.py --mode topography
```
Forces the use of topographic elevation colormap, ignoring any available satellite imagery. This mode uses synthetic terrain with realistic features.

### Command Line Options

```bash
python sim_visualize_tracks.py --help
```

Available options:
- `--mode {satellite,topography,auto}`: Choose visualization mode (default: auto)
- `--photo-zoom FLOAT`: Camera zoom factor for static photos (default: 1.0, higher values = more zoomed in)
- `--video-zoom FLOAT`: Camera zoom factor for video animations (default: 1.0, higher values = more zoomed in)
- `--intermediate-frames INT`: Number of intermediate frames between global time steps (default: 4, higher values = smoother animation)
- `--static-only`: Skip animation generation and only create static visualization
- `--topology-offset FLOAT`: Topography offset in meters - adjusts where the terrain surface appears relative to z=0 (default: 300.0, higher values = terrain appears lower)

### Examples

```bash
# Use satellite imagery with custom zoom levels and smooth animation
python sim_visualize_tracks.py --mode satellite --photo-zoom 1.5 --video-zoom 2.0 --intermediate-frames 8

# Use topography mode with closer camera for photos and very smooth animation
python sim_visualize_tracks.py --mode topography --photo-zoom 2.0 --intermediate-frames 10

# Auto-detect mode with wider view for videos and standard smoothness
python sim_visualize_tracks.py --mode auto --video-zoom 0.7 --intermediate-frames 4

# Static-only mode - skip animation for faster processing
python sim_visualize_tracks.py --mode topography --static-only --photo-zoom 1.5

# Adjust terrain positioning - fine-tune where objects at z=0 appear relative to terrain
python sim_visualize_tracks.py --mode topography --topology-offset 200 --photo-zoom 1.5
```

### Input Data Format

The trajectory data should be in JSON format with the following structure:

```json
{
    "track_history": [
        {
            "id": 1,
            "timesteps": [
                {
                    "time": 0.0,
                    "position": [0.0, 0.0, 100.0],
                    "velocity": [10.0, 5.0, -2.0]
                },
                {
                    "time": 1.0,
                    "position": [10.0, 5.0, 98.0],
                    "velocity": [12.0, 4.0, -1.5]
                }
            ]
        }
    ]
}
```

Where:
- `id`: Unique identifier for each trajectory
- `time`: Timestamp in seconds
- `position`: [East, North, Up] coordinates in meters
- `velocity`: [East, North, Up] velocity components in m/s

### Advanced Usage

#### Custom Visualization

```python
from sim_visualize_tracks import TrajectoryVisualizer

# Create visualizer
visualizer = TrajectoryVisualizer()

# Load your trajectory data
trajectories = visualizer.load_trajectories_from_json('your_data.json')

# Create static visualization
visualizer.create_static_visualization('output.png')

# Create interactive visualization
visualizer.create_interactive_visualization()

# Create animation
trajectory_meshes = visualizer.create_trajectory_meshes()
visualizer.animate_trajectories(
    trajectory_meshes, 
    fps=10, 
    save_path='animation.mp4'
)
```

#### Custom Topography

The system includes a `TopographyManager` class that can be extended to use real OpenTopography data:

```python
from sim_visualize_tracks import TopographyManager, TrajectoryVisualizer

# Create custom topography manager
topo_manager = TopographyManager()
# Add your OpenTopography API key if needed
topo_manager.api_key = "your_api_key"

# Create visualizer with custom topography
visualizer = TrajectoryVisualizer(topography_manager=topo_manager)
```

## Output Files

- `trajectory_static.png`: Static 3D visualization image
- `trajectory_animation.mp4`: Time-based animation video
- Interactive 3D window for real-time exploration

## Visualization Features

### Topography
- Synthetic terrain with realistic features
- Color-coded elevation mapping
- Smooth shading and lighting effects
- Extensible for real topographic data

### Trajectories
- Color-coded trajectory lines
- Velocity vectors at each point
- Trajectory ID labels
- Tube rendering for better visibility

### Animation
- Time-based progression
- Multiple trajectories active simultaneously
- Velocity vector animation
- Time display overlay

### Interactive Controls
- Mouse rotation, zoom, and pan
- Coordinate axes display
- Legend and information overlays
- Real-time interaction

## Technical Details

### Dependencies
- **PyVista**: 3D visualization and mesh processing
- **NumPy**: Numerical computations
- **Matplotlib**: Color mapping and plotting utilities
- **VTK**: Underlying 3D graphics engine
- **Requests**: HTTP requests for data download

### Coordinate System
The visualization uses the ENU (East-North-Up) coordinate system:
- **East**: Positive X-axis
- **North**: Positive Y-axis  
- **Up**: Positive Z-axis

### Performance
- Optimized for real-time interaction
- Efficient mesh processing
- Memory-conscious animation generation
- Scalable to large trajectory datasets

## Troubleshooting

### Common Issues

1. **PyVista installation issues**:
   ```bash
   pip install pyvista[all]
   ```

2. **VTK rendering problems**:
   - Update graphics drivers
   - Try different VTK backends

3. **Animation creation fails**:
   - Ensure ffmpeg is installed
   - Check file permissions for output directory

4. **Memory issues with large datasets**:
   - Reduce topography resolution
   - Use fewer trajectory points
   - Increase system memory

### WSL (Windows Subsystem for Linux) Issues

If you're running this visualization in WSL and experiencing display problems, the system includes automatic WSL detection and configuration. However, you may need to set up graphics support:

#### Quick Diagnosis

Run the WSL troubleshooting script:
```bash
python wsl_troubleshooting.py
```

This will check your WSL setup and provide specific solutions.

#### Common WSL Solutions

**For Windows 11 users:**
1. Enable WSLg (Windows 11 feature):
   ```bash
   wsl --update
   wsl --shutdown
   ```
2. Restart your computer
3. WSLg should provide native graphics support

**For Windows 10 users:**
1. Install VcXsrv: https://sourceforge.net/projects/vcxsrv/
2. Configure VcXsrv to allow connections from WSL
3. Set environment variables in your shell profile (~/.bashrc or ~/.zshrc):
   ```bash
   export DISPLAY=:0
   export LIBGL_ALWAYS_INDIRECT=1
   export MESA_GL_VERSION_OVERRIDE=3.3
   ```

**Alternative solutions:**
- Use the static visualization (`trajectory_static.png`) and animation (`trajectory_animation.mp4`) files
- Run the script on Windows directly (not WSL)
- Use WSL2 with GPU passthrough (advanced setup)

#### WSL-Specific Features

The visualization system automatically:
- Detects WSL environment
- Adjusts window sizes for WSL compatibility
- Uses WSL-optimized rendering settings
- Provides fallback options when display issues are detected
- Offers interactive prompts to continue or skip problematic visualizations

## Contributing

Feel free to extend the visualization suite with:
- Additional topography data sources
- New visualization styles
- Export formats
- Performance optimizations
- Additional analysis tools

## License

This project is open source and available under the MIT License.