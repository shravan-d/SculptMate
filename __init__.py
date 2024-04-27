bl_info = {
    "name": "SculptMate", 
    "description": "Generate a 3D Human Mesh from an image", 
    "author": "Shravan",
    "version": (0, 1),
    "blender": (3, 2, 0),
    "location": "Render Properties > SculptMate",
    "category": "3D View",
}

import bpy
from . import GUIPanel
import os
import sys
import subprocess

def install_dependencies():
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    python_executable = sys.executable
    try:
        import torch
    except ImportError:
        print('Installing required python dependencies. This should take a couple minutes')
        subprocess.check_call([python_executable, "-m", "pip", "install", "-r", requirements_file])

def register():
    install_dependencies()
    GUIPanel.register()
    
def unregister():
    GUIPanel.unregister()