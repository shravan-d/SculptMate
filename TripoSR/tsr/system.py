import os
from dataclasses import dataclass
from typing import List, Union
import numpy as np
import PIL.Image
import torch
import bpy
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

from .models.isosurface import MarchingCubeHelper
from .utils import (
    BaseModule,
    ImagePreprocessor,
    find_class,
    scale_tensor,
)
from .models.tokenizers.image import DINOSingleImageTokenizer
from .models.tokenizers.triplane import Triplane1DTokenizer
from .models.transformer.transformer_1d import Transformer1D
from .models.network_utils import TriplaneUpsampleNetwork
from .models.network_utils import NeRFMLP
from .models.nerf_renderer import TriplaneNeRFRenderer

class TSR(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int

        image_tokenizer_cls: str
        image_tokenizer: dict

        tokenizer_cls: str
        tokenizer: dict

        backbone_cls: str
        backbone: dict

        post_processor_cls: str
        post_processor: dict

        decoder_cls: str
        decoder: dict

        renderer_cls: str
        renderer: dict

    cfg: Config

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, config_name: str, weight_name: str
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, config_name)
            weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
        else:
            raise FileNotFoundError('Checkpoint directory given doesnt exist')

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        ckpt = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(ckpt)
        return model

    def configure(self):
        self.image_tokenizer = DINOSingleImageTokenizer(
            self.cfg.image_tokenizer
        )
        self.tokenizer = Triplane1DTokenizer(self.cfg.tokenizer)
        self.backbone = Transformer1D(self.cfg.backbone)
        self.post_processor = TriplaneUpsampleNetwork(
            self.cfg.post_processor
        )
        self.decoder = NeRFMLP(self.cfg.decoder)
        self.renderer = TriplaneNeRFRenderer(self.cfg.renderer)
        self.image_processor = ImagePreprocessor()
        self.isosurface_helper = None

    def forward(
        self,
        image: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.FloatTensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.FloatTensor],
        ],
        device: str,
    ) -> torch.FloatTensor:
        rgb_cond = self.image_processor(image, self.cfg.cond_image_size)[:, None].to(
            device
        )
        batch_size = rgb_cond.shape[0]

        input_image_tokens: torch.Tensor = self.image_tokenizer(
            rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1),
        )

        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1
        )

        tokens: torch.Tensor = self.tokenizer(batch_size)

        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
        )

        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        return scene_codes


    def set_marching_cubes_resolution(self, resolution: int):
        if (
            self.isosurface_helper is not None
            and self.isosurface_helper.resolution == resolution
        ):
            return
        self.isosurface_helper = MarchingCubeHelper(resolution)


    def import_obj_blender(self, verts, faces, vertex_colors=None, name='NewMesh'):
        mesh_data = bpy.data.meshes.new(name=name)
        mesh_data.from_pydata(verts, [], faces)
        new_object = bpy.data.objects.new(name=name, object_data=mesh_data)
        bpy.context.collection.objects.link(new_object)

        if vertex_colors is not None:
            if vertex_colors.shape[1] == 3:
                ones_column = np.ones((vertex_colors.shape[0], 1))
                vertex_colors = np.hstack((vertex_colors, ones_column))
            
            vertex_colors_name = f"{name}_VC"
            mesh_data.vertex_colors.new(name=vertex_colors_name)
            color_layer = mesh_data.vertex_colors[vertex_colors_name]

            # Assigning vertex colors
            for poly in mesh_data.polygons:
                for idx in poly.loop_indices:
                    loop_vert_index = mesh_data.loops[idx].vertex_index
                    color_layer.data[idx].color = vertex_colors[loop_vert_index]

            mat = bpy.data.materials.new(name="VertexColorMaterial")
            mesh_data.materials.append(mat)
                
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            for node in nodes:
                nodes.remove(node)

            output_node = nodes.new(type='ShaderNodeOutputMaterial')
            principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
            vertex_color_node = nodes.new(type='ShaderNodeVertexColor')

            vertex_color_node.layer_name = vertex_colors_name

            links.new(vertex_color_node.outputs['Color'], principled_node.inputs['Base Color'])
            links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

            principled_node.inputs['Roughness'].default_value = 1
            principled_node.inputs['IOR'].default_value = 1.00


    def extract_mesh(self, scene_codes, enable_texture = False, mesh_name='NewMesh', resolution: int = 256, threshold: float = 25.0):
        self.set_marching_cubes_resolution(resolution)
        for scene_code in scene_codes:
            with torch.no_grad():
                density = self.renderer.query_triplane(
                    self.decoder,
                    scale_tensor(
                        self.isosurface_helper.grid_vertices.to(scene_codes.device),
                        self.isosurface_helper.points_range,
                        (-self.renderer.cfg.radius, self.renderer.cfg.radius),
                    ),
                    scene_code,
                )["density_act"]
            v_pos, t_pos_idx = self.isosurface_helper(-(density - threshold))
            v_pos = scale_tensor(
                v_pos,
                self.isosurface_helper.points_range,
                (-self.renderer.cfg.radius, self.renderer.cfg.radius),
            )
            color = None
            if enable_texture:
                with torch.no_grad():
                    color = self.renderer.query_triplane(
                        self.decoder,
                        v_pos,
                        scene_code,
                    )["color"]
                color = color.cpu().numpy()
            
            self.import_obj_blender(v_pos.cpu().numpy(), t_pos_idx.cpu().numpy(), color, name=mesh_name)
