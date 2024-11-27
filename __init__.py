bl_info = {
    "name": "SculptMate", 
    "description": "Generate a 3D mesh from an image",
    "author": "Shravan",
    "version": (0, 5, 0),
    "blender": (3, 2, 0),
    "location": "Render Properties > SculptMate",
    "category": "3D View",
    "doc_url": "https://github.com/shravan-d/SculptMate#readme",
	"tracker_url": "https://github.com/shravan-d/SculptMate/issues",
}

import bpy
import os
import sys
import subprocess
import threading
import importlib
import urllib.request
from collections import namedtuple
from .utils import label_multiline
from . import addon_updater_ops

Dependency = namedtuple("Dependency", ["module", "package", "name"])
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Declare all modules that this add-on depends on, that may need to be installed. The package and (global) name can be
# set to None, if they are equal to the module name. See import_module and ensure_and_import_module for the explanation
# of the arguments. DO NOT use this to import other parts of your Python add-on, import them as usual with an
# "import" statement.
dependencies = (Dependency(module="numpy", package=None, name=None),
                Dependency(module="pillow", package=None, name=None),
                Dependency(module="torch", package=None, name=None),
                Dependency(module="onnxruntime", package=None, name=None),
                Dependency(module="omegaconf==2.3.0", package=None, name=None),
                Dependency(module="einops==0.7.0", package=None, name=None),
                Dependency(module="transformers==4.38.0", package=None, name=None),
                Dependency(module="opencv-python", package=None, name=None),
                Dependency(module="jsonschema", package=None, name=None),
                Dependency(module="jaxtyping", package=None, name=None),
                Dependency(module="gpytoolbox", package=None, name=None),
                Dependency(module="open_clip_torch==2.24.0", package=None, name=None),
                Dependency(module="scikit-image", package=None, name=None))


dependencies_installed = False
lean_checkpoint_found = False
fast_checkpoint_found = False

def import_module(module_name, global_name=None, reload=True):
    """
    Import a module.
    :param module_name: Module to import.
    :param global_name: (Optional) Name under which the module is imported. If None the module_name will be used.
       This allows to import under a different name with the same effect as e.g. "import numpy as np" where "np" is
       the global_name under which the module can be accessed.
    :raises: ImportError and ModuleNotFoundError
    """
    if global_name is None:
        global_name = module_name

    if global_name in globals():
        importlib.reload(globals()[global_name])
    else:
        # Attempt to import the module and assign it to globals dictionary. This allow to access the module under
        # the given name, just like the regular import would.
        globals()[global_name] = importlib.import_module(module_name)


def install_pip():
    """
    Installs pip if not already present. Please note that ensurepip.bootstrap() also calls pip, which adds the
    environment variable PIP_REQ_TRACKER. After ensurepip.bootstrap() finishes execution, the directory doesn't exist
    anymore. However, when subprocess is used to call pip, in order to install a package, the environment variables
    still contain PIP_REQ_TRACKER with the now nonexistent path. This is a problem since pip checks if PIP_REQ_TRACKER
    is set and if it is, attempts to use it as temp directory. This would result in an error because the
    directory can't be found. Therefore, PIP_REQ_TRACKER needs to be removed from environment variables.
    :return:
    """

    try:
        # Check if pip is already installed
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True)
    except subprocess.CalledProcessError:
        import ensurepip

        ensurepip.bootstrap()
        os.environ.pop("PIP_REQ_TRACKER", None)


def install_and_import_module(module_name, package_name=None, global_name=None):
    """
    Installs the package through pip and attempts to import the installed module.
    :param module_name: Module to import.
    :param package_name: (Optional) Name of the package that needs to be installed. If None it is assumed to be equal
       to the module_name.
    :param global_name: (Optional) Name under which the module is imported. If None the module_name will be used.
       This allows to import under a different name with the same effect as e.g. "import numpy as np" where "np" is
       the global_name under which the module can be accessed.
    :raises: subprocess.CalledProcessError and ImportError
    """
    if package_name is None:
        package_name = module_name

    if global_name is None:
        global_name = module_name

    # Blender disables the loading of user site-packages by default. However, pip will still check them to determine
    # if a dependency is already installed. This can cause problems if the packages is installed in the user
    # site-packages and pip deems the requirement satisfied, but Blender cannot import the package from the user
    # site-packages. Hence, the environment variable PYTHONNOUSERSITE is set to disallow pip from checking the user
    # site-packages. If the package is not already installed for Blender's Python interpreter, it will then try to.
    # The paths used by pip can be checked with `subprocess.run([bpy.app.binary_path_python, "-m", "site"], check=True)`

    # Create a copy of the environment variables and modify them for the subprocess call
    environ_copy = dict(os.environ)
    environ_copy["PYTHONNOUSERSITE"] = "1"

    if package_name == "torch":
        has_gpu = bpy.context.preferences.addons[__name__].preferences.has_gpu
        if has_gpu:
            print('Installing torch with GPU support')
            subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.3.0", "torchvision==0.18.0", "--index-url", "https://download.pytorch.org/whl/cu118"], check=True, env=environ_copy)
        else:
            subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.3.0", "torchvision==0.18.0", "--index-url", "https://download.pytorch.org/whl/cpu"], check=True, env=environ_copy)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True, env=environ_copy)

    # The installation succeeded, attempt to import the module again
    # import_module(module_name, global_name)

class DataStore:
    bpy.types.Scene.buttons_enabled = bpy.props.BoolProperty(default=True)
    bpy.types.Scene.installation_progress = bpy.props.IntProperty(default=-1)
    bpy.types.Scene.download_progress = bpy.props.IntProperty(default=-1)

    @classmethod
    def poll(cls, context):
        return True
    
    
class Warning_PT_panel(bpy.types.Panel):
    bl_label = "SculptMate"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    @classmethod
    def poll(self, context):
        checkpoint_found = lean_checkpoint_found or fast_checkpoint_found
        return not (dependencies_installed and checkpoint_found)

    def draw(self, context):
        layout = self.layout
        
        label_multiline(layout, text=f"Please install the missing dependencies for the {bl_info.get('name')} add-on.")
        label_multiline(layout, text=f"- You can do this via the add-on preferences (Edit > Preferences > Add-ons)")
        layout.separator()
        label_multiline(layout, text=f"This will install the required python dependencies and checkpoint files.")
        layout.separator()
        label_multiline(layout, text=f"If you encounter an error, Blender may have to be started with elevated permissions i.e. 'Run as Administrator'")

class Download_Lean_Model(DataStore, bpy.types.Operator):
    bl_idname = "example.download_checkpoint_lean"
    bl_label = "Get Checkpoint (Lean)"
    bl_description = ("Downloads the required model checkpoints required for generation. "
                      "Internet connection is required. Expected to take ~2 minutes.")
    bl_options = {"REGISTER", "INTERNAL"}

    @classmethod
    def poll(self, context):
        # Deactivate when checkpoints have been installed
        return not lean_checkpoint_found and context.scene.buttons_enabled
    
    def install_complete_callback(self):
        bpy.context.scene.download_progress = -1
        global lean_checkpoint_found
        lean_checkpoint_found = True

        if dependencies_installed:
            from . import GUIPanel
            GUIPanel.register()
    
    def install_error_callback(self):
        bpy.context.scene.download_progress = -2

    def execute(self, context):
        # Install dependencies in background thread so Blender doesn't hang
        thread = DownloadWorker(self, self.install_complete_callback, self.install_error_callback, context, 'Trippo')
        thread.start()

        return {"FINISHED"}

class Download_Fast_Model(DataStore, bpy.types.Operator):
    bl_idname = "example.download_checkpoint_fast"
    bl_label = "Get Checkpoint (Better Geometry)"
    bl_description = ("Downloads the required model checkpoints required for generation. "
                      "Internet connection is required. Expected to take ~3 minutes.")
    bl_options = {"REGISTER", "INTERNAL"}

    @classmethod
    def poll(self, context):
        # Deactivate when checkpoints have been installed
        return not fast_checkpoint_found and context.scene.buttons_enabled
    
    def install_complete_callback(self):
        bpy.context.scene.download_progress = -1
        global fast_checkpoint_found
        fast_checkpoint_found = True

        if dependencies_installed:
            from . import GUIPanel
            GUIPanel.register()
    
    def install_error_callback(self):
        bpy.context.scene.download_progress = -2

    def execute(self, context):
        # Install dependencies in background thread so Blender doesn't hang
        thread = DownloadWorker(self, self.install_complete_callback, self.install_error_callback, context, 'Fast')
        thread.start()

        return {"FINISHED"}
    
class DownloadWorker(threading.Thread):

    def __init__(self, parent_cls, finish_callback, error_callback, context, download_type):
        self.parent_cls = parent_cls
        self.finish_callback = finish_callback
        self.error_callback = error_callback
        self.context = context
        self.download_type = download_type
        threading.Thread.__init__(self)

    def run(self):
        self.context.scene.buttons_enabled = False
        try:
            self.context.scene.download_progress = 0
            
            if not os.path.isfile(ROOT_DIR + '/checkpoints/u2net.onnx'):
                os.makedirs(ROOT_DIR + '/checkpoints', exist_ok=True)
                urllib.request.urlretrieve("https://github.com/shravan-d/SculptMate/releases/download/v0.2/u2net.onnx", ROOT_DIR + "/checkpoints/u2net.onnx")
                
            if self.download_type == 'Trippo':
                urllib.request.urlretrieve("https://github.com/shravan-d/SculptMate/releases/download/v0.2/model.ckpt", ROOT_DIR + "/TripoSR/checkpoints/model.ckpt")
            elif self.download_type == 'Fast':
                urllib.request.urlretrieve("https://github.com/shravan-d/SculptMate/releases/download/v0.5/model.safetensors", ROOT_DIR + "/StableFast/checkpoints/model.safetensors")
            else:
                pass
        except Exception as err:
            print('[Download Error]', err)
            self.context.scene.buttons_enabled = True
            self.error_callback()
            return
        self.finish_callback()
        self.context.scene.buttons_enabled = True

class Install_dependencies(DataStore, bpy.types.Operator):
    bl_idname = "example.install_dependencies"
    bl_label = "Install dependencies"
    bl_description = ("Downloads and installs the required python packages for this add-on. "
                      "Internet connection is required. Expected to take ~2 minutes.")
    bl_options = {"REGISTER", "INTERNAL"}

    @classmethod
    def poll(self, context):
        # Deactivate when dependencies have been installed
        return not dependencies_installed and context.scene.buttons_enabled
    
    def install_complete_callback(self):
        bpy.context.scene.installation_progress = -1
        global dependencies_installed
        dependencies_installed = True
        
        if lean_checkpoint_found or fast_checkpoint_found:
            from . import GUIPanel
            GUIPanel.register()
    
    def install_error_callback(self):
        bpy.context.scene.installation_progress = -2

    def execute(self, context):
        install_pip()
        # Install dependencies in background thread so Blender doesn't hang
        thread = InstallationWorker(self, self.install_complete_callback, self.install_error_callback, context)
        thread.start()

        return {"FINISHED"}

class Reconfigure_Torch(DataStore, bpy.types.Operator):
    bl_idname = "example.reconfigure_torch"
    bl_label = "Reconfigure GPU Support"
    bl_description = ("If you installed without GPU support, this reconfigures the add-on to use GPU."
                      "Internet connection is required. Expected to take ~2 minutes.")
    bl_options = {"REGISTER", "INTERNAL"}

    @classmethod
    def poll(self, context):
        # Deactivate when dependencies have been installed
        return context.scene.buttons_enabled
    
    def install_complete_callback(self):
        bpy.context.scene.installation_progress = -1
        global dependencies_installed
        dependencies_installed = True
        
        if lean_checkpoint_found or fast_checkpoint_found:
            from . import GUIPanel
            GUIPanel.register()
    
    def install_error_callback(self):
        bpy.context.scene.installation_progress = -2

    def execute(self, context):
        # Install dependencies in background thread so Blender doesn't hang
        thread = TorchReconWorker(self, self.install_complete_callback, self.install_error_callback, context)
        thread.start()

        return {"FINISHED"}

class TorchReconWorker(threading.Thread):

    def __init__(self, parent_cls, finish_callback, error_callback, context):
        self.parent_cls = parent_cls
        self.finish_callback = finish_callback
        self.error_callback = error_callback
        self.context = context
        threading.Thread.__init__(self)

    def run(self):
        self.context.scene.buttons_enabled = False
        try:
            environ_copy = dict(os.environ)
            environ_copy["PYTHONNOUSERSITE"] = "1"
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch"], check=True, env=environ_copy)
            subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.3.0", "torchvision==0.18.0", "--index-url", "https://download.pytorch.org/whl/cu118"], check=True, env=environ_copy)

        except (subprocess.CalledProcessError, ImportError) as err:
            print('[Installation Error]', err)
            self.context.scene.buttons_enabled = True
            self.error_callback()
            return
        self.finish_callback()
        self.context.scene.buttons_enabled = True

class InstallationWorker(threading.Thread):

    def __init__(self, parent_cls, finish_callback, error_callback, context):
        self.parent_cls = parent_cls
        self.finish_callback = finish_callback
        self.error_callback = error_callback
        self.context = context
        threading.Thread.__init__(self)

    def run(self):
        self.context.scene.buttons_enabled = False
        try:
            for dependency in dependencies:
                self.context.scene.installation_progress += 1
                install_and_import_module(module_name=dependency.module,
                                          package_name=dependency.package,
                                          global_name=dependency.name)
        except (subprocess.CalledProcessError, ImportError) as err:
            print('[Installation Error]', err)
            self.context.scene.buttons_enabled = True
            self.error_callback()
            return
        self.finish_callback()
        self.context.scene.buttons_enabled = True


class MyPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__
    auto_check_update: bpy.props.BoolProperty(
		name="Auto-check for Update",
		description="If enabled, auto-check for updates using an interval",
		default=False) # type: ignore
    
    has_gpu: bpy.props.BoolProperty(
        name="Has GPU",
        description="Check this if your system has a GPU available",
        default=False,
    ) # type: ignore

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.column().prop(self, "has_gpu")
        row.column().separator()
        row.column().operator(Reconfigure_Torch.bl_idname)

        layout.operator(Install_dependencies.bl_idname, icon="CONSOLE")
        if context.scene.installation_progress == 0:
            layout.label(text='Installation starting.')
        if context.scene.installation_progress > 0:
            layout.label(text=f'Installation Progress: {int(100*context.scene.installation_progress / len(dependencies))}%')
        if context.scene.installation_progress == -2:
            layout.label(text='Installation Failed. Please check system console for details.')

        row = layout.row()
        col = row.column()
        col.operator(Download_Lean_Model.bl_idname)
        col = row.column()
        col.operator(Download_Fast_Model.bl_idname)
        if context.scene.download_progress == 0:
            layout.label(text='Downloading')
        if context.scene.download_progress == -2:
            layout.label(text='Download Failed. Please check system console for details.')

        addon_updater_ops.update_settings_ui_condensed(self, context)


preference_classes = (Warning_PT_panel,
                      Install_dependencies,
                      Download_Lean_Model,
                      Download_Fast_Model,
                      Reconfigure_Torch,
                      MyPreferences)


def register():
    global dependencies_installed
    dependencies_installed = False
    global lean_checkpoint_found
    lean_checkpoint_found = False
    global fast_checkpoint_found
    fast_checkpoint_found = False
    addon_updater_ops.register(bl_info)

    for cls in preference_classes:
        bpy.utils.register_class(cls)

    if os.path.isfile(ROOT_DIR + '/TripoSR/checkpoints/model.ckpt'):
        lean_checkpoint_found = True
    if os.path.isfile(ROOT_DIR + '/StableFast/checkpoints/model.safetensors'):
        fast_checkpoint_found = True

    try:
        # for dependency in dependencies:
        #     import_module(module_name=dependency.module, global_name=dependency.name)
        import PIL
        import torch
        import torchvision
        import numpy
        import jsonschema
        import skimage
        import onnxruntime
        import omegaconf
        import einops
        import cv2
        import transformers
        import jaxtyping
        import gpytoolbox
        import open_clip
        dependencies_installed = True
    except ModuleNotFoundError as err:
        print('[Missing Module Error]', err)
        # Don't register other panels, operators etc.
        return

    if not fast_checkpoint_found and not lean_checkpoint_found:
        print('[Missing Checkpoints Error] Please download checkpoints from the Preferences and ensure they are placed in the right directories.')
        return

    import faulthandler
    faulthandler.enable()
    from . import GUIPanel
    GUIPanel.register()


def unregister():
    for cls in preference_classes:
        bpy.utils.unregister_class(cls)

    if dependencies_installed:
        from . import GUIPanel
        GUIPanel.unregister()