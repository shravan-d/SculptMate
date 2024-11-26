import numpy as np
import torch

class tb_float2:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return tb_float2(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return tb_float2(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar):
        return tb_float2(self.x * scalar, self.y * scalar)

    def __repr__(self):
        return f"tb_float2(x={self.x}, y={self.y})"


class tb_float3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, other):
        return tb_float3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return tb_float3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __add__(self, other):
        return tb_float3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __repr__(self):
        return f"tb_float3(x={self.x}, y={self.y}, z={self.z})"


class AABB:
    def __init__(self):
        self.min = tb_float2(np.finfo(np.float32).max, np.finfo(np.float32).max)
        self.max = tb_float2(np.finfo(np.float32).min, np.finfo(np.float32).min)

    def grow(self, p):
        """Expands the bounding box to include point p."""
        self.min.x = min(self.min.x, p.x)
        self.min.y = min(self.min.y, p.y)
        self.max.x = max(self.max.x, p.x)
        self.max.y = max(self.max.y, p.y)

    def overlaps(self, other):
        """Checks if this bounding box overlaps another."""
        if isinstance(other, AABB):
            return (
                self.min.x <= other.max.x
                and self.max.x >= other.min.x
                and self.min.y <= other.max.y
                and self.max.y >= other.min.y
            )
        elif isinstance(other, tb_float2):
            return (
                other.x >= self.min.x
                and other.x <= self.max.x
                and other.y >= self.min.y
                and other.y <= self.max.y
            )

    def invalidate(self):
        self.min = tb_float2(np.finfo(np.float32).max, np.finfo(np.float32).max)
        self.max = tb_float2(np.finfo(np.float32).min, np.finfo(np.float32).min)

    def area(self):
        """Returns the area of the bounding box."""
        return (self.max.x - self.min.x) * (self.max.y - self.min.y)

class BVHNode:
    def __init__(self, start=0, end=0):
        self.start = start
        self.end = end
        self.left = None
        self.right = None
        self.bbox = AABB()

    def is_leaf(self):
        return self.left is None and self.right is None


def calc_mean(a, b, c):
    return (a + b + c) / 3

class Triangle:
    def __init__(self, v0: tb_float2, v1: tb_float2, v2: tb_float2, idx: int):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.centroid = tb_float3(
            calc_mean(v0.x, v1.x, v2.x),
            calc_mean(v0.y, v1.y, v2.y)
        )
        self.index = idx

def barycentric_coordinates(p, v0, v1, v2):
    """Computes the barycentric coordinates of point p with respect to triangle (v0, v1, v2)."""
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    pv0 = p - v0

    d00 = v0v1.x * v0v1.x + v0v1.y * v0v1.y
    d01 = v0v1.x * v0v2.x + v0v1.y * v0v2.y
    d11 = v0v2.x * v0v2.x + v0v2.y * v0v2.y
    d20 = pv0.x * v0v1.x + pv0.y * v0v1.y
    d21 = pv0.x * v0v2.x + pv0.y * v0v2.y

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return u, v, w

def rasterize_cpu(vertices, indices, resolution):
    """Rasterizes UVs using a BVH."""
    bvh = BVH()
    bvh.build(vertices, indices)

    width, height = resolution, resolution
    output = np.zeros((height, width, 4), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            point = tb_float2(x / width, 1.0 - y / height)
            result = bvh.intersect(point)

            if result:
                tri_idx, u, v, w = result
                output[y, x] = [u, v, w, tri_idx]
            else:
                output[y, x] = [0, 0, 0, -1]

    return output

def rasterize(vertices, indices, resolution, device='cuda', batch_size=2):
    """
    Rasterizes UVs using a BVH or directly with barycentric coordinates.
    Optimized with PyTorch for GPU acceleration.
    Args:
        vertices (torch.Tensor): Vertex positions of shape (n, 2).
        indices (torch.Tensor): Triangle indices of shape (m, 3).
        resolution (int): Image resolution (width and height).
        device (str): 'cuda' for GPU or 'cpu' for CPU.
    Returns:
        torch.Tensor: Rasterized barycentric coordinates and triangle indices of shape (resolution, resolution, 4).
    """
    # Prepare pixel grid
    width, height = resolution, resolution
    pixels_x = torch.linspace(0, 1, width, device=device)
    pixels_y = torch.linspace(0, 1, height, device=device)

    # Move vertices and indices to GPU if necessary
    vertices = vertices.to(device)
    indices = indices.to(device)

    # Get triangle vertices
    v0 = vertices[indices[:, 0]]  # (m, 2)
    v1 = vertices[indices[:, 1]]  # (m, 2)
    v2 = vertices[indices[:, 2]]  # (m, 2)

    # Calculate edge vectors
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    d00 = (v0v1 * v0v1).sum(dim=-1)  # (m,)
    d01 = (v0v1 * v0v2).sum(dim=-1)  # (m,)
    d11 = (v0v2 * v0v2).sum(dim=-1)  # (m,)
    denom = d00 * d11 - d01 * d01  # (m,)
    
    result = torch.zeros((height, width, 4), device=device)
    
    for start_y in range(0, height, batch_size):
        end_y = min(start_y + batch_size, height)
        batch_pixels_y = pixels_y[start_y:end_y]
        grid_x, grid_y = torch.meshgrid(pixels_x, batch_pixels_y, indexing='ij')
        pixels = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=-1)  # (batch_width*batch_height, 2)
    
        # Compute barycentric coordinates for all triangles and all pixels
        pv0 = pixels.unsqueeze(1) - v0  # (width*height, m, 2)
        d20 = (pv0 * v0v1.unsqueeze(0)).sum(dim=-1)  # (width*height, m)
        d21 = (pv0 * v0v2.unsqueeze(0)).sum(dim=-1)  # (width*height, m)

        v = (d11 * d20 - d01 * d21) / denom  # (width*height, m)
        w = (d00 * d21 - d01 * d20) / denom  # (width*height, m)
        u = 1 - v - w  # (width*height, m)

        # Determine if pixels are inside the triangle
        inside = (u >= 0) & (v >= 0) & (w >= 0)  # (width*height, m)

        # Select the first triangle that covers each pixel
        pixel_mask = inside.any(dim=1)  # (width*height,)
        triangle_idx = torch.argmax(inside.float(), dim=1)  # (width*height,)

        # Assign barycentric coordinates to output
        uvw = torch.stack((u, v, w), dim=-1)  # (width*height, m, 3)
        batch_result = torch.zeros((pixels.shape[0], 4), device=device)
        batch_result[pixel_mask, :3] = uvw[pixel_mask, triangle_idx[pixel_mask]]
        batch_result[pixel_mask, 3] = triangle_idx[pixel_mask].float()

        batch_result = batch_result.view(end_y - start_y, width, 4)
        result[start_y:end_y, :, :] = batch_result

    return result


def interpolate_cpu(attr, indices, rast):
    height, width = rast.shape[:2]
    output = np.zeros((height, width, 3), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            bary = rast[y, x]
            if bary[3] < 0:
                continue

            tri_idx = int(bary[3])
            u, v, w = bary[:3]
            v1, v2, v3 = indices[tri_idx]
            output[y, x] = attr[v1] * u + attr[v2] * v + attr[v3] * w

    return output


def interpolate(attr, indices, rast, device='cuda'):
    """
    Interpolates attributes over a rasterized image using barycentric coordinates.
    Optimized with PyTorch for GPU acceleration.
    Args:
        attr (torch.Tensor): Vertex attributes of shape (n, k).
        indices (torch.Tensor): Triangle indices of shape (m, 3).
        rast (torch.Tensor): Rasterized output of shape (resolution, resolution, 4).
        device (str): 'cuda' for GPU or 'cpu' for CPU.
    Returns:
        torch.Tensor: Interpolated attributes of shape (resolution, resolution, k).
    """
    height, width, _ = rast.shape
    rast = rast.view(-1, 4)  # (resolution*resolution, 4)

    # Move attributes and indices to GPU if necessary
    attr = attr.to(device)
    indices = indices.to(device)

    # Get barycentric coordinates and triangle indices
    uvw = rast[:, :3]  # (resolution*resolution, 3)
    triangle_idx = rast[:, 3].long()  # (resolution*resolution,)

    # Interpolate attributes
    interpolated_attr = torch.zeros((rast.shape[0], attr.shape[1]), device=device)
    valid = triangle_idx >= 0
    if valid.any():
        v0 = attr[indices[triangle_idx[valid], 0]]  # (valid_pixels, k)
        v1 = attr[indices[triangle_idx[valid], 1]]  # (valid_pixels, k)
        v2 = attr[indices[triangle_idx[valid], 2]]  # (valid_pixels, k)
        bary = uvw[valid]  # (valid_pixels, 3)
        interpolated_attr[valid] = v0 * bary[:, 0:1] + v1 * bary[:, 1:2] + v2 * bary[:, 2:3]

    return interpolated_attr.view(height, width, -1)



class BVH:
    def __init__(self):
        self.nodes = []
        self.triangles = []
        self.triangle_indices = []

    def build(self, vertices, indices):
        """Builds a BVH for a given set of triangles."""
        self.triangles = [
            Triangle(
                tb_float2(*vertices[indices[i][0]]),
                tb_float2(*vertices[indices[i][1]]),
                tb_float2(*vertices[indices[i][2]]),
                i,
            )
            for i in range(len(indices))
        ]
        self.triangle_indices = list(range(len(self.triangles)))

        root = BVHNode(0, len(self.triangles))
        self.nodes.append(root)

        self._split_node(root)

    def _split_node(self, node):
        """Recursively splits BVH nodes to create the hierarchy."""
        node.bbox = AABB()
        for i in range(node.start, node.end):
            tri = self.triangles[self.triangle_indices[i]]
            node.bbox.grow(tri.v0)
            node.bbox.grow(tri.v1)
            node.bbox.grow(tri.v2)

        if node.end - node.start <= 2:  # Leaf node condition
            return

        # Determine split axis
        axis = (
            0
            if (node.bbox.max.x - node.bbox.min.x)
            > (node.bbox.max.y - node.bbox.min.y)
            else 1
        )

        # Sort triangle indices along the chosen axis
        self.triangle_indices[node.start : node.end] = sorted(
            self.triangle_indices[node.start : node.end],
            key=lambda i: getattr(self.triangles[i].centroid, "x" if axis == 0 else "y"),
        )

        mid = (node.start + node.end) // 2

        # Create child nodes
        node.left = BVHNode(node.start, mid)
        node.right = BVHNode(mid, node.end)
        self.nodes.append(node.left)
        self.nodes.append(node.right)

        self._split_node(node.left)
        self._split_node(node.right)

    def intersect(self, point):
        """Performs a BVH intersection for the given point."""
        stack = [self.nodes[0]]
        while stack:
            node = stack.pop()

            if not node.bbox.overlaps(point):
                continue

            if node.is_leaf():
                for i in range(node.start, node.end):
                    tri = self.triangles[self.triangle_indices[i]]
                    u, v, w = barycentric_coordinates(point, tri.v0, tri.v1, tri.v2)
                    if u >= 0 and v >= 0 and w >= 0:
                        return tri.index, u, v, w
            else:
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
        return None
