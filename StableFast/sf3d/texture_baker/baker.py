import torch
import torch.nn as nn
from torch import Tensor
import os
import ctypes
import numpy as np

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
        dll_path = os.path.join(os.path.dirname(__file__), "texture_baker.dll")
        dll = ctypes.CDLL(dll_path)

        dll.rasterize_cpu.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int), ctypes.c_size_t,
            ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_float)
        ]
        dll.rasterize_cpu.restype = None
        
        nv = uv.size(0)
        nf = face_indices.size(0)

        uv_numpy = uv.cpu().numpy().flatten()
        indices_numpy = face_indices.to(torch.int32).cpu().numpy().flatten()
        rast_result = np.zeros((bake_resolution * bake_resolution, 4), dtype=np.float32).flatten()

        dll.rasterize_cpu(
            uv_numpy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), nv,
            indices_numpy.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), nf,
            bake_resolution,
            rast_result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        rast_result = rast_result.reshape(bake_resolution, bake_resolution, 4)

        return torch.from_numpy(rast_result).to(device)

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
        bake_resolution: int,
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
        dll_path = os.path.join(os.path.dirname(__file__), "texture_baker.dll")
        baker_dll = ctypes.CDLL(dll_path)

        baker_dll.interpolate_cpu.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_float)
        ]
        baker_dll.interpolate_cpu.restype = None
        
        nv = attr.size(0)
        nf = face_indices.size(0)

        attr_numpy = attr.cpu().numpy().flatten()
        indices_numpy = face_indices.to(torch.int32).cpu().numpy().flatten()
        rast_numpy = rast.cpu().numpy().flatten()
        inter_result = np.zeros((bake_resolution * bake_resolution, 3), dtype=np.float32).flatten()

        baker_dll.interpolate_cpu(
            attr_numpy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), nv,
            indices_numpy.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), nf,
            rast_numpy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), bake_resolution,
            inter_result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        inter_result = inter_result.reshape(bake_resolution, bake_resolution, 3)

        return torch.from_numpy(inter_result).to(device)

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
        return self.interpolate(attr, rast, face_indices, uv, bake_resolution, device)
