#!/usr/bin/env python3
"""
Test script for the trajectory visualization suite with real-world coordinates.
"""

import json
from sim_visualize_tracks import TrajectoryVisualizer

def test_real_world():
    """Test the visualization suite with real-world coordinates."""
    print("🧪 Testing Trajectory Visualization Suite - Real World Coordinates")
    print("=" * 70)
    
    # Arizona reference origin (lat, lon, height) - matches topography data
    ref_origin = (32.9203654, -113.9303655, 0.0)
    
    # Create trajectory data spanning several kilometers (ENU meters)
    real_world_data = {
        "track_history": [
            {
                "id": 1,
                "timesteps": [
                    {
                        "time": 0.0,
                        "position": [0.0, 0.0, 1000.0],
                        "velocity": [100.0, 50.0, -20.0]
                    },
                    {
                        "time": 10.0,
                        "position": [1000.0, 500.0, 800.0],
                        "velocity": [120.0, 40.0, -15.0]
                    },
                    {
                        "time": 20.0,
                        "position": [2200.0, 900.0, 650.0],
                        "velocity": [150.0, 30.0, -10.0]
                    }
                ]
            },
            {
                "id": 2,
                "timesteps": [
                    {
                        "time": 5.0,
                        "position": [500.0, 1000.0, 1500.0],
                        "velocity": [80.0, 120.0, -30.0]
                    },
                    {
                        "time": 15.0,
                        "position": [1300.0, 2200.0, 1200.0],
                        "velocity": [100.0, 110.0, -25.0]
                    },
                    {
                        "time": 25.0,
                        "position": [2300.0, 3300.0, 950.0],
                        "velocity": [120.0, 100.0, -20.0]
                    }
                ]
            }
        ]
    }
    
    # Save to temporary file
    with open('temp_real_world.json', 'w') as f:
        json.dump(real_world_data, f, indent=2)
    
    # Create visualizer with real-world reference origin
    visualizer = TrajectoryVisualizer(ref_origin=ref_origin)
    
    # Load trajectories
    print(f"Loading trajectory data with reference origin: {ref_origin}")
    try:
        trajectories = visualizer.load_trajectories_from_json('temp_real_world.json')
        print(f"✅ Successfully loaded {len(trajectories)} trajectories")
        
        # Print trajectory info
        for trajectory in trajectories:
            print(f"  Trajectory {trajectory.id}: {len(trajectory.points)} points")
            if trajectory.points:
                print(f"    Time range: {trajectory.points[0].time:.1f}s - {trajectory.points[-1].time:.1f}s")
                print(f"    Position range: {trajectory.points[0].position} to {trajectory.points[-1].position}")
        
        # Test bounds calculation
        bounds = visualizer.calculate_bounds()
        print(f"✅ Calculated bounds: {bounds}")
        
        # Calculate area in km² (approximate)
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        area_km2 = x_range * y_range * 111 * 111  # Rough conversion to km²
        print(f"✅ Approximate area: {area_km2:.0f} km²")
        
        # Test trajectory mesh creation
        trajectory_meshes = visualizer.create_trajectory_meshes()
        print(f"✅ Created {len(trajectory_meshes)} trajectory meshes")
        
        # Test static visualization (save to file)
        print("Creating static visualization with real topography...")
        visualizer.create_static_visualization('test_real_world_static.png')
        print("✅ Static visualization created successfully")
        
        print("\n🎉 Real-world test passed! The visualization suite is working correctly.")
        print("\nTo run the full visualization suite:")
        print("  python sim_visualize_tracks.py")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temporary file
        import os
        if os.path.exists('temp_real_world.json'):
            os.remove('temp_real_world.json')

if __name__ == "__main__":
    test_real_world() 