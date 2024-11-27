import torch
import os
from contextlib import nullcontext
from .sf3d.system import SF3D

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class Fast3DGenerator():
    def __init__(self, device):
        self.checkpoint_dir = ROOT_DIR + '/checkpoints/'
        self.texture_resolution = 1024
        self.image_path = ''
        self.device = device
        self.model = None

    def initiate_model(self):
        if self.model is None:
            try:
                self.model = SF3D.from_pretrained(
                    self.checkpoint_dir,
                    config_name="config.yaml",
                    weight_name="model.safetensors",
                    device=self.device
                )
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print('[Model Dos Initialization Error]', e)
                return 2
            return 0

    def generate_mesh(self, input_image, input_name=None, 
                      remesh_option='triangle', 
                      texture_resolution=512, 
                      vertex_simplification_factor='high',
                      enable_texture=True):
        if self.model is None:
            return 1
        # try:
        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16
            ) if "cuda" == self.device.type else nullcontext():
                mesh, glob_dict = self.model.run_image(
                    input_image,
                    bake_resolution=texture_resolution,
                    remesh=remesh_option,
                    vertex_simplification_factor=vertex_simplification_factor,
                    enable_texture=enable_texture
                )
        if mesh is None:
            raise Exception('Mesh shape was zero')

        self.model.import_mesh_blender(mesh, input_name)
        return 0
        # except Exception as e:
        #     print('[Generation Error]', e)
        #     return 2
