import torch
import torch.nn as nn
from torch import Tensor
from .common import rasterize, interpolate

class TextureBaker(nn.Module):
    def __init__(self):
        super().__init__()

    def rasterize(
        self,
        uv: Tensor,
        face_indices: Tensor,
        bake_resolution: int,
        device
    ) -> Tensor:
        """
        Rasterize the UV coordinates to a barycentric coordinates
        & Triangle idxs texture map

        Args:
            uv (Tensor, num_vertices 2, float): UV coordinates of the mesh
            face_indices (Tensor, num_faces 3, int): Face indices of the mesh
            bake_resolution (int): Resolution of the bake

        Returns:
            Tensor, bake_resolution bake_resolution 4, float: Rasterized map
        """
        return rasterize(
            uv, face_indices.to(torch.int32), bake_resolution, device=device
        )

    def get_mask(self, rast: Tensor) -> Tensor:
        """
        Get the occupancy mask from the rasterized map

        Args:
            rast (Tensor, bake_resolution bake_resolution 4, float): Rasterized map

        Returns:
            Tensor, bake_resolution bake_resolution, bool: Mask
        """
        return rast[..., -1] >= 0

    def interpolate(
        self,
        attr: Tensor,
        rast: Tensor,
        face_indices: Tensor,
        device
    ) -> Tensor:
        """
        Interpolate the attributes using the rasterized map

        Args:
            attr (Tensor, num_vertices 3, float): Attributes of the mesh
            rast (Tensor, bake_resolution bake_resolution 4, float): Rasterized map
            face_indices (Tensor, num_faces 3, int): Face indices of the mesh
            uv (Tensor, num_vertices 2, float): UV coordinates of the mesh

        Returns:
            Tensor, bake_resolution bake_resolution 3, float: Interpolated attributes
        """
        return interpolate(
            attr, face_indices.to(torch.int32), rast, device=device
        )

    def forward(
        self,
        attr: Tensor,
        uv: Tensor,
        face_indices: Tensor,
        bake_resolution: int,
        device
    ) -> Tensor:
        """
        Bake the texture

        Args:
            attr (Tensor, num_vertices 3, float): Attributes of the mesh
            uv (Tensor, num_vertices 2, float): UV coordinates of the mesh
            face_indices (Tensor, num_faces 3, int): Face indices of the mesh
            bake_resolution (int): Resolution of the bake

        Returns:
            Tensor, bake_resolution bake_resolution 3, float: Baked texture
        """
        rast = self.rasterize(uv, face_indices, bake_resolution, device)
        return self.interpolate(attr, rast, face_indices, uv, device)
