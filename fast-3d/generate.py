import torch
import os
from contextlib import nullcontext
from PIL import Image
from sf3d.system import SF3D
from sf3d.utils import get_device, remove_background, resize_foreground

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
                model = SF3D.from_pretrained(
                    self.checkpoint_dir,
                    config_name="config.yaml",
                    weight_name="model.safetensors",
                )
                model.to(self.device)
                model.eval()
            except Exception as e:
                print('[Model Dos Initialization Error]', e)
                return 2
            return 0

    def generate_mesh(self, input_image, input_name=None, remesh_option='triangle', texture_resolution=1024, target_vertex_count=-1):
        if self.model is None:
            return 1
        try:
            with torch.no_grad():
                with torch.autocast(
                    device_type=self.device, dtype=torch.float16
                ) if "cuda" in self.device else nullcontext():
                    mesh, glob_dict = self.model.run_image(
                        input_image,
                        bake_resolution=texture_resolution,
                        remesh=remesh_option,
                        vertex_count=target_vertex_count,
                    )
            if mesh is None:
                raise Exception('Mesh shape was zero')

            self.model.import_mesh_blender(mesh, input_name)
            return 0
        except Exception as e:
            print('[Generation Error]', e)
            return 2
