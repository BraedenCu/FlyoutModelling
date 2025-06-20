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
python sim_visualize_tracks.py
```

This will:
1. Load trajectory data from `input.py`
2. Create a static visualization saved as `trajectory_static.png`
3. Generate an animation saved as `trajectory_animation.mp4`
4. Open an interactive 3D visualization window

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

### Performance Tips

- Use lower resolution topography for faster rendering
- Reduce animation frame rate for quicker video generation
- Close other applications to free up GPU memory
- Use SSD storage for faster file I/O

## Contributing

Feel free to extend the visualization suite with:
- Additional topography data sources
- New visualization styles
- Export formats
- Performance optimizations
- Additional analysis tools

## License

This project is open source and available under the MIT License.