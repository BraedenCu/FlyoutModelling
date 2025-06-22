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

# Copy the rest of your existing classes and functions here...
# (This is just the configuration part - you'll need to copy the rest from the original file) 