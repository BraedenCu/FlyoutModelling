#!/usr/bin/env python3
"""
Test script for the trajectory visualization suite.
"""

import json
from sim_visualize_tracks import TrajectoryVisualizer

def test_visualization():
    """Test the visualization suite with sample data."""
    print("🧪 Testing Trajectory Visualization Suite")
    print("=" * 50)
    
    # Create visualizer
    visualizer = TrajectoryVisualizer()
    
    # Load trajectories
    print("Loading trajectory data...")
    try:
        trajectories = visualizer.load_trajectories_from_json('input.py')
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
        
        # Test trajectory mesh creation
        trajectory_meshes = visualizer.create_trajectory_meshes()
        print(f"✅ Created {len(trajectory_meshes)} trajectory meshes")
        
        # Test static visualization (save to file)
        print("Creating static visualization...")
        visualizer.create_static_visualization('test_static.png')
        print("✅ Static visualization created successfully")
        
        print("\n🎉 All tests passed! The visualization suite is working correctly.")
        print("\nTo run the full visualization suite:")
        print("  python sim_visualize_tracks.py")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualization() 