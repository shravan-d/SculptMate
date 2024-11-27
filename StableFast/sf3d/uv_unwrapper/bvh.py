import numpy as np
from typing import List, Tuple, Deque
from collections import deque

# Import helper classes and methods
from .common import UVFloat2, triangle_centroid_2d, cross_3d, normalize, EPSILON
from .intersect import triangle_triangle_intersection, triangle_aabb_intersection

BINS = 8

# TODO: Should this be 2D or 3D?
class AABB:
    def __init__(self):
        self.min = UVFloat2(np.inf, np.inf)
        self.max = UVFloat2(-np.inf, -np.inf)

    def grow(self, point: UVFloat2):
        self.min = UVFloat2(
            min(self.min.x, point.x),
            min(self.min.y, point.y),
        )
        self.max = UVFloat2(
            max(self.max.x, point.x),
            max(self.max.y, point.y),
        )

    def grow_aabb(self, other: 'AABB'):
        self.grow(other.min)
        self.grow(other.max)

    def overlaps(self, triangle: 'Triangle') -> bool:
        return triangle_aabb_intersection(self.min, self.max, triangle.a, triangle.b, triangle.c)

    def area(self) -> float:
        extent = self.max - self.min
        return extent.x * extent.y

    def invalidate(self):
        self.min = UVFloat2(np.inf, np.inf)
        self.max = UVFloat2(-np.inf, -np.inf)


class BVHNode:
    def __init__(self):
        self.bbox = AABB()
        self.start = 0
        self.end = 0
        self.left = -1
        self.right = -1

    def num_triangles(self) -> int:
        return self.end - self.start

    def is_leaf(self) -> bool:
        return self.left == -1 and self.right == -1

    def calculate_node_cost(self) -> float:
        return self.num_triangles() * self.bbox.area()


class Triangle:
    def __init__(self, v0: UVFloat2, v1: UVFloat2, v2: UVFloat2):
        self.a = v0
        self.b = v1
        self.c = v2
        self.centroid = triangle_centroid_2d(v0, v1, v2)

    def overlaps(self, other: 'Triangle') -> bool:
        return triangle_triangle_intersection(self.a, self.b, self.c, other.a, other.b, other.c)

    def get_normal(self) -> UVFloat2:
        u = self.b - self.a
        v = self.c - self.a
        return normalize(cross_3d(u, v))

    def is_degenerate(self) -> bool:
        u = self.a - self.b
        v = self.a - self.c
        cr = cross_3d(u, v)
        return abs(cr.x) < EPSILON and abs(cr.y) < EPSILON and abs(cr.z) < EPSILON


class BVH:
    def __init__(self, triangles: List[Triangle] = None, indices: List[int] = None):
        self.triangles = triangles
        self.indices = indices
        self.tri_count = len(indices) if indices is not None else 0
        self.triIdx = np.arange(self.tri_count)
        self.nodes = [BVHNode() for _ in range(self.tri_count * 2 + 64)]
        if triangles is not None:
            self.nodes_used = 2
            self.initialize_root()

    def initialize_root(self):
        root = self.nodes[0]
        root.start = 0
        root.end = self.tri_count
        root_centroid_bounds = self.update_node_bounds(0)
        self.subdivide(0, root_centroid_bounds)

    def update_node_bounds(self, node_idx: int):
        centroid_bounds = AABB()
        node = self.nodes[node_idx]
        node.bbox.invalidate()
        centroid_bounds.invalidate()

        for i in range(node.start, node.end):
            tri = self.triangles[self.triIdx[i]]
            node.bbox.grow(tri.a)
            node.bbox.grow(tri.b)
            node.bbox.grow(tri.c)
            centroid_bounds.grow(tri.centroid)

        return centroid_bounds

    def subdivide(self, root_idx: int, root_centroid_bounds: AABB):
        node_queue: Deque[Tuple[int, AABB]] = deque([(root_idx, root_centroid_bounds)])

        while node_queue:
            node_idx, centroid_bounds = node_queue.popleft()
            node = self.nodes[node_idx]

            axis, split_pos, cost = self.find_best_split_plane(node, centroid_bounds)
            if cost >= node.calculate_node_cost():
                node.left = node.right = -1
                continue

            mid = self.partition_triangles(node, axis, split_pos, centroid_bounds)

            left_child_idx = self.nodes_used
            self.nodes_used += 1
            right_child_idx = self.nodes_used
            self.nodes_used += 1

            node.left = left_child_idx
            node.right = right_child_idx

            self.nodes[left_child_idx].start = node.start
            self.nodes[left_child_idx].end = mid
            self.nodes[right_child_idx].start = mid
            self.nodes[right_child_idx].end = node.end

            left_bounds = self.update_node_bounds(left_child_idx)
            node_queue.append((left_child_idx, left_bounds))

            right_bounds = self.update_node_bounds(right_child_idx)
            node_queue.append((right_child_idx, right_bounds))

    def partition_triangles(
        self, node: BVHNode, axis: int, split_pos: int, centroid_bounds: AABB
    ) -> int:
        scale = 8 / (centroid_bounds.max[axis] - centroid_bounds.min[axis])
        i, j = node.start, node.end - 1

        while i <= j:
            centroid = self.triangles[self.triIdx[i]].centroid[axis]
            bin_idx = int((centroid - centroid_bounds.min[axis]) * scale)
            if bin_idx < split_pos:
                i += 1
            else:
                self.triIdx[i], self.triIdx[j] = self.triIdx[j], self.triIdx[i]
                j -= 1

        return i

    def find_best_split_plane(self, node: BVHNode, centroid_bounds: AABB):
        best_cost = float("inf")
        best_axis, best_pos = -1, -1

        for axis in range(2):
            bounds_min, bounds_max = centroid_bounds.min[axis], centroid_bounds.max[axis]
            if bounds_min == bounds_max:
                continue

            bins = [{'bounds': AABB(), 'triCount': 0} for _ in range(8)]
            for i in range(node.start, node.end):
                tri = self.triangles[self.triIdx[i]]
                bin_idx = int((tri.centroid[axis] - bounds_min) * BINS / (bounds_max - bounds_min))
                bins[min(bin_idx, BINS - 1)]['bounds'].grow(tri.a)
                bins[min(bin_idx, BINS - 1)]['bounds'].grow(tri.b)
                bins[min(bin_idx, BINS - 1)]['bounds'].grow(tri.c)
                bins[min(bin_idx, BINS - 1)]['triCount'] += 1
            
            left_box, right_box = AABB(), AABB()
            left_sum, right_sum = 0, 0
            left_area, right_area = [None] * BINS, [None] * BINS
            for i in range(BINS - 1):
                left_sum += bins[i]['triCount']
                left_box.grow_aabb(bins[i]['bounds'])
                left_area[i] = left_sum * left_box.area()
                right_sum += bins[BINS - 1 - i]['triCount']
                right_box.grow_aabb(bins[BINS - 1 - i]['bounds'])
                right_area[i] = right_sum * right_box.area()

            for i in range(BINS - 1):
                cost = left_area[i] + right_area[i]
                if cost < best_cost:
                    best_cost = cost
                    best_axis = axis
                    best_pos = i + 1

        return best_axis, best_pos, best_cost

    def intersect(self, triangle: Triangle) -> List[int]:
        intersected = []
        stack = [0]

        while stack:
            node_idx = stack.pop()
            node = self.nodes[node_idx]

            if node.is_leaf():
                for i in range(node.start, node.end):
                    tri = self.triangles[self.triIdx[i]]
                    if tri != triangle and triangle.overlaps(tri):
                        intersected.append(self.indices[self.triIdx[i]])
            else:
                if self.nodes[node.right].bbox.overlaps(triangle):
                    stack.append(node.right)
                if self.nodes[node.left].bbox.overlaps(triangle):
                    stack.append(node.left)

        return intersected
