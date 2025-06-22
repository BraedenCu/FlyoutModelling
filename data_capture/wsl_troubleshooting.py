#!/usr/bin/env python3
"""
WSL Visualization Troubleshooting Script
Helps diagnose and fix common WSL graphics issues for PyVista visualization.
"""

import os
import subprocess
import sys
import platform

def check_wsl():
    """Check if running in WSL."""
    try:
        with open('/proc/version', 'r') as f:
            content = f.read().lower()
            if 'microsoft' in content:
                return True
    except:
        pass
    return False

def check_x_server():
    """Check if X server is running and accessible."""
    print("Checking X server status...")
    
    # Check if DISPLAY is set
    display = os.environ.get('DISPLAY')
    if not display:
        print("   DISPLAY environment variable not set")
        return False
    
    print(f"   DISPLAY set to: {display}")
    
    # Try to connect to X server
    try:
        result = subprocess.run(['xset', 'q'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   X server is running and accessible")
            return True
        else:
            print("   X server is not responding")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   X server not found or not responding")
        return False

def check_graphics():
    """Check graphics capabilities."""
    print("Checking graphics capabilities...")
    
    # Check for OpenGL
    try:
        result = subprocess.run(['glxinfo'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Parse OpenGL version
            for line in result.stdout.split('\n'):
                if 'OpenGL version string' in line:
                    version = line.split(':')[1].strip()
                    print(f"   OpenGL version: {version}")
                    return True
            print("   OpenGL available but version not found")
            return True
        else:
            print("   Could not get OpenGL information")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   glxinfo not available")
        return False

def check_pyvista():
    """Check PyVista installation and capabilities."""
    print("Checking PyVista...")
    
    try:
        import pyvista as pv
        print(f"   PyVista version: {pv.__version__}")
        
        # Test basic rendering
        try:
            # Create a simple plot
            plotter = pv.Plotter(off_screen=True)
            sphere = pv.Sphere()
            plotter.add_mesh(sphere)
            plotter.screenshot('test_render.png')
            plotter.close()
            
            # Clean up test file
            if os.path.exists('test_render.png'):
                os.remove('test_render.png')
            
            print("   PyVista rendering test successful")
            return True
        except Exception as e:
            print(f"   PyVista rendering test failed: {e}")
            return False
            
    except ImportError:
        print("   PyVista not installed")
        return False

def check_wslg():
    """Check if running in WSLg (Windows 11 WSL)."""
    print("Checking WSLg support...")
    
    try:
        # Check if running in WSL
        with open('/proc/version', 'r') as f:
            content = f.read().lower()
            if 'microsoft' in content:
                # Check for WSLg
                if 'wslg' in content:
                    print("   WSLg detected - native graphics support available")
                    return True
                else:
                    print("   WSLg not detected - may need X server")
                    return False
            else:
                print("   Not running in WSL")
                return False
    except Exception as e:
        print("   Could not determine WSL version")
        return False

def test_visualization_module():
    """Test the visualization module."""
    print("Testing visualization module...")
    
    try:
        # Import the visualization module
        import sys
        sys.path.append('..')
        from sim_visualize_tracks import TrajectoryVisualizer
        
        print("   Visualization module imports successfully")
        
        # Test basic functionality
        try:
            visualizer = TrajectoryVisualizer()
            print("   Static visualization test successful")
            return True
        except Exception as e:
            print(f"   Static visualization test failed: {e}")
            return False
            
    except Exception as e:
        print(f"   Visualization module test failed: {e}")
        return False

def main():
    """Main troubleshooting function."""
    print("WSL Graphics Troubleshooting Tool")
    print("=" * 40)
    
    # Check if running in WSL
    try:
        with open('/proc/version', 'r') as f:
            content = f.read().lower()
            if 'microsoft' not in content:
                print("Not running in WSL. This script is for WSL users only.")
                return
    except:
        print("Not running in WSL. This script is for WSL users only.")
        return
    
    print("Running in WSL")
    print()
    
    # Run all checks
    display_ok = check_x_server()
    print()
    
    graphics_ok = check_graphics()
    print()
    
    pyvista_ok = check_pyvista()
    print()
    
    wslg_available = check_wslg()
    print()
    
    viz_ok = test_visualization_module()
    print()
    
    # Summary
    print("Summary:")
    print(f"   Display: {'OK' if display_ok else 'FAILED'}")
    print(f"   Graphics: {'OK' if graphics_ok else 'FAILED'}")
    print(f"   PyVista: {'OK' if pyvista_ok else 'FAILED'}")
    print(f"   WSLg: {'OK' if wslg_available else 'FAILED'}")
    print(f"   Visualization: {'OK' if viz_ok else 'FAILED'}")
    
    # Provide solutions
    if not all([display_ok, graphics_ok, pyvista_ok, wslg_available, viz_ok]):
        print("\nSome issues detected. Follow the solutions above.")
        
        if not wslg_available:
            print("\nWSLg Solutions:")
            print("1. Update WSL: wsl --update")
            print("2. Restart WSL: wsl --shutdown")
            print("3. Make sure you're running Windows 11 with WSLg support")
        
        if not display_ok and not wslg_available:
            print("\nX Server Solutions:")
            print("1. Install VcXsrv or Xming on Windows")
            print("2. Set DISPLAY=:0 in WSL")
            print("3. Configure Windows firewall to allow X server")
        
        if not pyvista_ok:
            print("\nPyVista Solutions:")
            print("1. Install PyVista: pip install pyvista")
            print("2. Install additional dependencies: pip install pyvista[all]")
        
        if not viz_ok:
            print("\nVisualization Solutions:")
            print("1. Check that all required files are present")
            print("2. Verify Python path includes the project directory")
            print("3. Check for missing dependencies")
    else:
        print("\nAll checks passed! Your WSL setup should work for visualization.")

if __name__ == "__main__":
    main() 