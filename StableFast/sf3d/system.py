import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from jaxtyping import Float
from PIL import Image
from safetensors.torch import load_model
from torch import Tensor
import bpy

from .models.isosurface import MarchingTetrahedraHelper
from .models.mesh import Mesh
from .models.utils import (
    BaseModule,
    ImageProcessor,
    convert_data,
    dilate_fill,
    dot,
    find_class,
    float32_to_uint8_np,
    normalize,
    scale_tensor,
)

from .models.tokenizers.image import DINOV2SingleImageTokenizer
from .models.tokenizers.triplane import TriplaneLearnablePositionalEmbedding
from .models.transformers.backbone import TwoStreamInterleaveTransformer
from .models.network import PixelShuffleUpsampleNetwork, MaterialMLP
from .models.global_estimator.multi_head_estimator import MultiHeadEstimator
from .models.image_estimator.clip_based_estimator import ClipBasedHeadEstimator
from .models.camera import LinearCameraEmbedder


from .utils import create_intrinsic_from_fov_deg, default_cond_c2w, get_device
from .texture_baker.baker import TextureBaker

class SF3D(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int
        isosurface_resolution: int
        isosurface_threshold: float = 10.0
        radius: float = 1.0
        background_color: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
        default_fovy_deg: float = 40.0
        default_distance: float = 1.6

        camera_embedder_cls: str = ""
        camera_embedder: dict = field(default_factory=dict)

        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        decoder_cls: str = ""
        decoder: dict = field(default_factory=dict)

        image_estimator_cls: str = ""
        image_estimator: dict = field(default_factory=dict)

        global_estimator_cls: str = ""
        global_estimator: dict = field(default_factory=dict)

    cfg: Config

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, config_name: str, weight_name: str, device: str
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, config_name)
            weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
        else:
            raise FileNotFoundError('Checkpoint directory given doesnt exist')
        cls.device = device
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        load_model(model, weight_path)
        return model

    def configure(self):
        self.image_tokenizer = DINOV2SingleImageTokenizer(
            self.cfg.image_tokenizer
        )
        self.tokenizer = TriplaneLearnablePositionalEmbedding(self.cfg.tokenizer)
        self.camera_embedder = LinearCameraEmbedder(
            self.cfg.camera_embedder
        )
        self.backbone = TwoStreamInterleaveTransformer(self.cfg.backbone)
        self.post_processor = PixelShuffleUpsampleNetwork(
            self.cfg.post_processor
        )
        self.decoder = MaterialMLP(self.cfg.decoder)
        self.image_estimator = ClipBasedHeadEstimator(
            self.cfg.image_estimator
        )
        self.global_estimator = MultiHeadEstimator(
            self.cfg.global_estimator
        )

        self.bbox: Float[Tensor, "2 3"] # type: ignore
        self.register_buffer(
            "bbox",
            torch.as_tensor(
                [
                    [-self.cfg.radius, -self.cfg.radius, -self.cfg.radius],
                    [self.cfg.radius, self.cfg.radius, self.cfg.radius],
                ],
                dtype=torch.float32,
            ),
        )
        self.isosurface_helper = MarchingTetrahedraHelper(
            self.cfg.isosurface_resolution,
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "load",
                "tets",
                f"{self.cfg.isosurface_resolution}_tets.npz",
            ),
        )

        self.baker = TextureBaker()
        self.image_processor = ImageProcessor()

    def triplane_to_meshes(
        self, triplanes: Float[Tensor, "B 3 Cp Hp Wp"] # type: ignore
    ) -> list[Mesh]:
        meshes = []
        for i in range(triplanes.shape[0]):
            triplane = triplanes[i]
            grid_vertices = scale_tensor(
                self.isosurface_helper.grid_vertices.to(triplanes.device),
                self.isosurface_helper.points_range,
                self.bbox,
            )

            values = self.query_triplane(grid_vertices, triplane)
            decoded = self.decoder(values, include=["vertex_offset", "density"])
            sdf = decoded["density"] - self.cfg.isosurface_threshold

            deform = decoded["vertex_offset"].squeeze(0)

            mesh: Mesh = self.isosurface_helper(
                sdf.view(-1, 1), deform.view(-1, 3) if deform is not None else None
            )
            mesh.v_pos = scale_tensor(
                mesh.v_pos, self.isosurface_helper.points_range, self.bbox
            )

            meshes.append(mesh)

        return meshes

    def query_triplane(
        self,
        positions: Float[Tensor, "*B N 3"], # type: ignore
        triplanes: Float[Tensor, "*B 3 Cp Hp Wp"], # type: ignore
    ) -> Float[Tensor, "*B N F"]: # type: ignore
        batched = positions.ndim == 3
        if not batched:
            # no batch dimension
            triplanes = triplanes[None, ...]
            positions = positions[None, ...]
        assert triplanes.ndim == 5 and positions.ndim == 3

        positions = scale_tensor(
            positions, (-self.cfg.radius, self.cfg.radius), (-1, 1)
        )

        indices2D: Float[Tensor, "B 3 N 2"] = torch.stack( # type: ignore
            (positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]),
            dim=-3,
        ).to(triplanes.dtype)
        out: Float[Tensor, "B3 Cp 1 N"] = F.grid_sample( # type: ignore
            rearrange(triplanes, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np=3).float(),
            rearrange(indices2D, "B Np N Nd -> (B Np) () N Nd", Np=3).float(),
            align_corners=True,
            mode="bilinear",
        )
        out = rearrange(out, "(B Np) Cp () N -> B N (Np Cp)", Np=3)

        return out

    def get_scene_codes(self, batch) -> Float[Tensor, "B 3 C H W"]: # type: ignore
        # if batch[rgb_cond] is only one view, add a view dimension
        if len(batch["rgb_cond"].shape) == 4:
            batch["rgb_cond"] = batch["rgb_cond"].unsqueeze(1)
            batch["mask_cond"] = batch["mask_cond"].unsqueeze(1)
            batch["c2w_cond"] = batch["c2w_cond"].unsqueeze(1)
            batch["intrinsic_cond"] = batch["intrinsic_cond"].unsqueeze(1)
            batch["intrinsic_normed_cond"] = batch["intrinsic_normed_cond"].unsqueeze(1)

        batch_size, n_input_views = batch["rgb_cond"].shape[:2]

        camera_embeds: Optional[Float[Tensor, "B Nv Cc"]] # type: ignore
        camera_embeds = self.camera_embedder(**batch)

        input_image_tokens: Float[Tensor, "B Nv Cit Nit"] = self.image_tokenizer( # type: ignore
            rearrange(batch["rgb_cond"], "B Nv H W C -> B Nv C H W"),
            modulation_cond=camera_embeds,
        )

        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=n_input_views
        )

        tokens: Float[Tensor, "B Ct Nt"] = self.tokenizer(batch_size) # type: ignore

        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
            modulation_cond=None,
        )

        direct_codes = self.tokenizer.detokenize(tokens)
        scene_codes = self.post_processor(direct_codes)
        return scene_codes, direct_codes

    def run_image(
        self,
        image: Union[Image.Image, List[Image.Image]],
        bake_resolution: int,
        remesh: Literal["none", "triangle", "quad"] = "none",
        vertex_simplification_factor: Literal['high', 'medium', 'low'] = 'high',
        estimate_illumination: bool = False,
        enable_texture: bool = True
    ):
        if isinstance(image, list):
            rgb_cond = []
            mask_cond = []
            for img in image:
                mask, rgb = self.prepare_image(img)
                mask_cond.append(mask)
                rgb_cond.append(rgb)
            rgb_cond = torch.stack(rgb_cond, 0)
            mask_cond = torch.stack(mask_cond, 0)
            batch_size = rgb_cond.shape[0]
        else:
            mask_cond, rgb_cond = self.prepare_image(image)
            batch_size = 1

        c2w_cond = default_cond_c2w(self.cfg.default_distance).to(self.device)
        intrinsic, intrinsic_normed_cond = create_intrinsic_from_fov_deg(
            self.cfg.default_fovy_deg,
            self.cfg.cond_image_size,
            self.cfg.cond_image_size,
        )

        batch = {
            "rgb_cond": rgb_cond,
            "mask_cond": mask_cond,
            "c2w_cond": c2w_cond.view(1, 1, 4, 4).repeat(batch_size, 1, 1, 1),
            "intrinsic_cond": intrinsic.to(self.device)
            .view(1, 1, 3, 3)
            .repeat(batch_size, 1, 1, 1),
            "intrinsic_normed_cond": intrinsic_normed_cond.to(self.device)
            .view(1, 1, 3, 3)
            .repeat(batch_size, 1, 1, 1),
        }

        meshes, global_dict = self.generate_mesh(
            batch, bake_resolution, remesh, vertex_simplification_factor, estimate_illumination, enable_texture
        )
        if batch_size == 1:
            return meshes[0], global_dict
        else:
            return meshes, global_dict

    def prepare_image(self, image):
        if image.mode != "RGBA":
            raise ValueError("Image must be in RGBA mode")
        img_cond = (
            torch.from_numpy(
                np.asarray(
                    image.resize((self.cfg.cond_image_size, self.cfg.cond_image_size))
                ).astype(np.float32)
                / 255.0
            )
            .float()
            .clip(0, 1)
            .to(self.device)
        )
        mask_cond = img_cond[:, :, -1:]
        rgb_cond = torch.lerp(
            torch.tensor(self.cfg.background_color, device=self.device)[None, None, :],
            img_cond[:, :, :3],
            mask_cond,
        )

        return mask_cond, rgb_cond

    def generate_mesh(
        self,
        batch,
        bake_resolution: int,
        remesh: Literal["none", "triangle", "quad"] = "none",
        vertex_simplification_factor: Literal['high', 'medium', 'low'] = 'high',
        estimate_illumination: bool = False,
        enable_texture: bool = True
    ):
        batch["rgb_cond"] = self.image_processor(
            batch["rgb_cond"], self.cfg.cond_image_size
        )
        batch["mask_cond"] = self.image_processor(
            batch["mask_cond"], self.cfg.cond_image_size
        )
        scene_codes, non_postprocessed_codes = self.get_scene_codes(batch)

        global_dict = {}
        if self.image_estimator is not None:
            global_dict.update(
                self.image_estimator(batch["rgb_cond"] * batch["mask_cond"])
            )
        if self.global_estimator is not None and estimate_illumination:
            global_dict.update(self.global_estimator(non_postprocessed_codes))

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type, enabled=False
            ) if "cuda" == self.device.type else nullcontext():
                meshes = self.triplane_to_meshes(scene_codes)
                
                rets = []
                for i, mesh in enumerate(meshes):
                    # Check for empty mesh
                    if mesh.v_pos.shape[0] == 0:
                        rets.append(None)
                        continue
                    
                    if vertex_simplification_factor == 'high':
                        vertex_count = round(0.75 * mesh.v_pos.shape[0])
                    elif vertex_simplification_factor == 'med':
                        vertex_count = round(0.4 * mesh.v_pos.shape[0])
                    else:
                        vertex_count = round(0.1 * mesh.v_pos.shape[0])

                    if remesh == "triangle":
                        mesh = mesh.triangle_remesh(triangle_vertex_count=vertex_count)
                    elif remesh == "quad":
                        mesh = mesh.quad_remesh(quad_vertex_count=vertex_count)

                    mesh.unwrap_uv()
                    if enable_texture:
                        # Build textures
                        rast = self.baker.rasterize(
                            mesh.v_tex, mesh.t_pos_idx, bake_resolution, self.device
                        )
                        bake_mask = self.baker.get_mask(rast)

                        pos_bake = self.baker.interpolate(
                            mesh.v_pos,
                            rast,
                            mesh.t_pos_idx,
                            bake_resolution,
                            self.device
                        )
                        gb_pos = pos_bake[bake_mask]

                        tri_query = self.query_triplane(gb_pos, scene_codes[i])[0]
                        decoded = self.decoder(
                            tri_query, exclude=["density", "vertex_offset"]
                        )

                        nrm = self.baker.interpolate(
                            mesh.v_nrm,
                            rast,
                            mesh.t_pos_idx,
                            bake_resolution,
                            self.device
                        )

                        gb_nrm = F.normalize(nrm[bake_mask], dim=-1)
                        decoded["normal"] = gb_nrm

                        # Check if any keys in global_dict start with decoded_
                        for k, v in global_dict.items():
                            if k.startswith("decoder_"):
                                decoded[k.replace("decoder_", "")] = v[i]

                        mat_out = {
                            "albedo": decoded["features"],
                            "roughness": decoded["roughness"],
                            "metallic": decoded["metallic"],
                            "normal": normalize(decoded["perturb_normal"]),
                            "bump": None,
                        }


                        for k, v in mat_out.items():
                            if v is None:
                                continue
                            if v.shape[0] == 1:
                                # Skip and directly add a single value
                                mat_out[k] = v[0]
                            else:
                                f = torch.zeros(
                                    bake_resolution,
                                    bake_resolution,
                                    v.shape[-1],
                                    dtype=v.dtype,
                                    device=v.device,
                                )
                                if v.shape == f.shape:
                                    continue
                                if k == "normal":
                                    # Use un-normalized tangents here so that larger smaller tris
                                    # Don't effect the tangents that much
                                    tng = self.baker.interpolate(
                                        mesh.v_tng,
                                        rast,
                                        mesh.t_pos_idx,
                                        bake_resolution,
                                        self.device
                                    )
                                    gb_tng = tng[bake_mask]
                                    gb_tng = F.normalize(gb_tng, dim=-1)
                                    gb_btng = F.normalize(
                                        torch.cross(gb_tng, gb_nrm, dim=-1), dim=-1
                                    )
                                    normal = F.normalize(mat_out["normal"], dim=-1)

                                    bump = torch.cat(
                                        # Check if we have to flip some things
                                        (
                                            dot(normal, gb_tng),
                                            dot(normal, gb_btng),
                                            dot(normal, gb_nrm).clip(
                                                0.3, 1
                                            ),  # Never go below 0.3. This would indicate a flipped (or close to one) normal
                                        ),
                                        -1,
                                    )
                                    bump = (bump * 0.5 + 0.5).clamp(0, 1)

                                    f[bake_mask] = bump.view(-1, 3)
                                    mat_out["bump"] = f
                                else:
                                    f[bake_mask] = v.view(-1, v.shape[-1])
                                    mat_out[k] = f

                        def uv_padding(arr):
                            if arr.ndim == 1:
                                return arr
                            return (
                                dilate_fill(
                                    arr.permute(2, 0, 1)[None, ...].contiguous(),
                                    bake_mask.unsqueeze(0).unsqueeze(0),
                                    iterations=bake_resolution // 150,
                                )
                                .squeeze(0)
                                .permute(1, 2, 0)
                                .contiguous()
                            )

                        basecolor_tex = Image.fromarray(
                            float32_to_uint8_np(convert_data(uv_padding(mat_out["albedo"])))
                        ).convert("RGBA")
                        basecolor_tex.format = "JPEG"

                        metallic = mat_out["metallic"].squeeze().cpu().item()
                        roughness = mat_out["roughness"].squeeze().cpu().item()

                        if "bump" in mat_out and mat_out["bump"] is not None:
                            bump_np = convert_data(uv_padding(mat_out["bump"]))
                            bump_up = np.ones_like(bump_np)
                            bump_up[..., :2] = 0.5
                            bump_up[..., 2:] = 1
                            bump_tex = Image.fromarray(
                                float32_to_uint8_np(
                                    bump_np,
                                    dither=True,
                                    # Do not dither if something is perfectly flat
                                    dither_mask=np.all(
                                        bump_np == bump_up, axis=-1, keepdims=True
                                    ).astype(np.float32),
                                )
                            ).convert("RGBA")
                            bump_tex.format = (
                                "JPEG"  # PNG would be better but the assets are larger
                            )
                        else:
                            bump_tex = None

                        verts_np = convert_data(mesh.v_pos)
                        faces = convert_data(mesh.t_pos_idx)
                        uvs = convert_data(mesh.v_tex)

                        rets.append({
                            'vertices': verts_np,
                            'faces': faces,
                            'uvs': uvs,
                            'basecolor_tex': basecolor_tex,
                            'bump_tex': bump_tex,
                            'roughness': roughness,
                            'metallic': metallic
                        })
                    else:
                        verts_np = convert_data(mesh.v_pos)
                        faces = convert_data(mesh.t_pos_idx)
                        uvs = convert_data(mesh.v_tex)

                        rets.append({
                            'vertices': verts_np,
                            'faces': faces,
                            'uvs': uvs,
                            'basecolor_tex': None,
                            'bump_tex': None,
                            'roughness': None,
                            'metallic': None
                        })

        return rets, global_dict

    def import_mesh_blender(self, mesh, mesh_name="GeneratedMesh"):  
        mesh_data = bpy.data.meshes.new(mesh_name)
        mesh_data.from_pydata(mesh['vertices'], [], mesh['faces'])
        obj = bpy.data.objects.new(name=mesh_name, object_data=mesh_data)
        bpy.context.collection.objects.link(obj)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # UV setup
        if mesh['uvs'] is not None:
            mesh_data.uv_layers.new(name="UVMap")
            uv_layer = mesh_data.uv_layers.active.data
            flattened_uvs = [uv for face in mesh['faces'] for uv in mesh['uvs'][face]]
            for i, loop in enumerate(mesh_data.loops):
                uv_layer[i].uv = flattened_uvs[i]

        # Material setup
        material = bpy.data.materials.new(name="PBRMaterial")
        material.use_nodes = True
        obj.data.materials.append(material)

        # Get nodes and links
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()  # Clear default nodes

        # Create Principled BSDF shader
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        output = nodes.new(type="ShaderNodeOutputMaterial")
        # output.location = (400, 0)
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

        # Set base color texture
        if mesh['basecolor_tex']:
            tex_image = nodes.new('ShaderNodeTexImage')
            image_data = np.array(mesh['basecolor_tex'])
            
            # Flip the texture vertically
            image_data = np.flip(image_data, axis=0)

            blender_image = bpy.data.images.new("BaseColor", width=mesh['basecolor_tex'].width, height=mesh['basecolor_tex'].height)
            blender_image.pixels = image_data.flatten() / 255.0
            tex_image.image = blender_image
            # tex_image.location = (-300, 200)
            links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])

        # Set roughness and metallic factors
        if mesh['roughness']:
            bsdf.inputs['Roughness'].default_value = mesh['roughness']
        if mesh['metallic']:
            bsdf.inputs['Metallic'].default_value = mesh['metallic']

        # Set normal map
        if mesh['bump_tex']:
            normal_map = nodes.new('ShaderNodeTexImage')
            image_data = np.array(mesh['bump_tex'])
            image_data = np.flip(image_data, axis=0)
            blender_image = bpy.data.images.new("Bump", width=mesh['bump_tex'].width, height=mesh['bump_tex'].height)
            blender_image.pixels = image_data.flatten() / 255.0
            normal_map.image = blender_image
            normal_map.image.colorspace_settings.name = 'Non-Color'
            # normal_map.location = (-300, -200)

            normal_map_node = nodes.new('ShaderNodeNormalMap')
            # normal_map_node.location = (-100, -200)
            links.new(normal_map.outputs['Color'], normal_map_node.inputs['Color'])
            links.new(normal_map_node.outputs['Normal'], bsdf.inputs['Normal'])
