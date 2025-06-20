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

def check_display():
    """Check display configuration."""
    print("🔍 Checking display configuration...")
    
    # Check DISPLAY environment variable
    display = os.environ.get('DISPLAY', '')
    print(f"   DISPLAY: {display}")
    
    # Check if X server is accessible
    try:
        result = subprocess.run(['xset', 'q'], capture_output=True, timeout=5)
        if result.returncode == 0:
            print("   ✅ X server is running and accessible")
            return True
        else:
            print("   ❌ X server is not responding")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   ❌ X server not found or not responding")
        return False

def check_graphics_drivers():
    """Check graphics driver information."""
    print("🔍 Checking graphics drivers...")
    
    try:
        # Check OpenGL info
        result = subprocess.run(['glxinfo', '-B'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'OpenGL renderer' in line:
                    print(f"   OpenGL Renderer: {line.split(':')[1].strip()}")
                elif 'OpenGL version' in line:
                    print(f"   OpenGL Version: {line.split(':')[1].strip()}")
            return True
        else:
            print("   ❌ Could not get OpenGL information")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   ❌ glxinfo not available")
        return False

def check_pyvista():
    """Test PyVista installation and basic functionality."""
    print("🔍 Testing PyVista...")
    
    try:
        import pyvista as pv
        print(f"   ✅ PyVista version: {pv.__version__}")
        
        # Test basic PyVista functionality
        try:
            # Create a simple plotter
            plotter = pv.Plotter(off_screen=True)
            sphere = pv.Sphere()
            plotter.add_mesh(sphere)
            
            # Try to render (this should work even without display)
            plotter.screenshot('test_render.png')
            print("   ✅ PyVista rendering test successful")
            
            # Clean up
            os.remove('test_render.png')
            return True
            
        except Exception as e:
            print(f"   ❌ PyVista rendering test failed: {e}")
            return False
            
    except ImportError:
        print("   ❌ PyVista not installed")
        return False

def check_windows_version():
    """Check Windows version for WSLg compatibility."""
    print("🔍 Checking Windows version...")
    
    try:
        # Try to get Windows version from WSL
        result = subprocess.run(['uname', '-r'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            kernel_version = result.stdout.strip()
            print(f"   WSL Kernel: {kernel_version}")
            
            # Check if WSLg is available (Windows 11 feature)
            if 'microsoft' in kernel_version and 'WSLg' in kernel_version:
                print("   ✅ WSLg detected - native graphics support available")
                return True
            else:
                print("   ⚠️ WSLg not detected - may need X server")
                return False
    except:
        print("   ❌ Could not determine WSL version")
        return False

def provide_solutions():
    """Provide solutions based on detected issues."""
    print("\n🔧 Solutions for WSL visualization issues:")
    print("=" * 50)
    
    is_wsl = check_wsl()
    if not is_wsl:
        print("Not running in WSL - no WSL-specific solutions needed.")
        return
    
    display_ok = check_display()
    pyvista_ok = check_pyvista()
    wslg_available = check_windows_version()
    
    print("\n📋 Recommended solutions:")
    
    if not display_ok:
        print("\n1. For Windows 11 users:")
        print("   - Enable WSLg: wsl --update")
        print("   - Restart WSL: wsl --shutdown")
        print("   - Restart your computer")
        
        print("\n2. For Windows 10 users:")
        print("   - Install VcXsrv: https://sourceforge.net/projects/vcxsrv/")
        print("   - Configure VcXsrv to allow connections from WSL")
        print("   - Set DISPLAY=:0 in your WSL environment")
        
        print("\n3. Alternative solutions:")
        print("   - Use the static visualization (trajectory_static.png)")
        print("   - Use the animation (trajectory_animation.mp4)")
        print("   - Run the script on Windows directly (not WSL)")
    
    if not pyvista_ok:
        print("\n4. PyVista issues:")
        print("   - Reinstall PyVista: pip install --upgrade pyvista")
        print("   - Install additional dependencies: pip install vtk")
    
    print("\n5. Environment setup:")
    print("   Add these to your ~/.bashrc or ~/.zshrc:")
    print("   export DISPLAY=:0")
    print("   export LIBGL_ALWAYS_INDIRECT=1")
    print("   export MESA_GL_VERSION_OVERRIDE=3.3")

def test_visualization():
    """Test if the visualization script works."""
    print("\n🧪 Testing visualization script...")
    
    try:
        # Import and test the main visualization module
        from sim_visualize_tracks import TrajectoryVisualizer
        
        # Create a minimal test
        visualizer = TrajectoryVisualizer()
        print("   ✅ Visualization module imports successfully")
        
        # Test static visualization creation
        try:
            visualizer.create_static_visualization('test_static.png')
            print("   ✅ Static visualization test successful")
            os.remove('test_static.png')
        except Exception as e:
            print(f"   ❌ Static visualization test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Visualization module test failed: {e}")
        return False

def main():
    """Main troubleshooting function."""
    print("🔧 WSL Visualization Troubleshooting")
    print("=" * 40)
    
    # Check if running in WSL
    if not check_wsl():
        print("❌ Not running in WSL. This script is for WSL users only.")
        return
    
    print("✅ Running in WSL")
    
    # Run all checks
    display_ok = check_display()
    graphics_ok = check_graphics_drivers()
    pyvista_ok = check_pyvista()
    wslg_available = check_windows_version()
    
    # Test visualization
    viz_ok = test_visualization()
    
    # Provide solutions
    provide_solutions()
    
    # Summary
    print("\n📊 Summary:")
    print(f"   Display: {'✅' if display_ok else '❌'}")
    print(f"   Graphics: {'✅' if graphics_ok else '❌'}")
    print(f"   PyVista: {'✅' if pyvista_ok else '❌'}")
    print(f"   WSLg: {'✅' if wslg_available else '❌'}")
    print(f"   Visualization: {'✅' if viz_ok else '❌'}")
    
    if all([display_ok, pyvista_ok, viz_ok]):
        print("\n🎉 All checks passed! Visualization should work.")
    else:
        print("\n⚠️ Some issues detected. Follow the solutions above.")

if __name__ == "__main__":
    main() 