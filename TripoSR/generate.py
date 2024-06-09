import torch
from .tsr.system import TSR
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class TripoGenerator():
    def __init__(self, device):
        self.checkpoint_dir = ROOT_DIR + '/checkpoints/'
        self.chunk_size = 8192
        self.image_path = ''
        self.mc_resolution = 256
        self.device = device
        self.model = None

    def initiate_model(self):
        if self.model is None:
            try:
                self.model = TSR.from_pretrained(
                    self.checkpoint_dir,
                    config_name="config.yaml",
                    weight_name="model.ckpt",
                )
                self.model.renderer.set_chunk_size(self.chunk_size)
                self.model.to(self.device)
            except Exception as e:
                print('[Model Dos Initialization Error]', e)
                return 2
            return 0

    def generate_mesh(self, input_image, input_name=None, enable_texture=False):
        if self.model is None:
            return 1
        try:
            with torch.no_grad():
                scene_codes = self.model([input_image], device=self.device)

            self.model.extract_mesh(scene_codes, resolution=self.mc_resolution, mesh_name=input_name, enable_texture=enable_texture)
            return 0
        except Exception as e:
            print('[Generation Error]', e)
            return 2
