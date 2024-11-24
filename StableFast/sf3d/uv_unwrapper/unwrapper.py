import numpy as np
from typing import List, Set
from .bvh import BVH, Triangle, UVFloat3

def create_bvhs(bvhs: List[BVH], triangles: List[Triangle], triangle_per_face: List[Set[int]],
                num_faces: int, start: int, end: int):
    for i in range(start, end):
        num_triangles = len(triangle_per_face[i])
        triangles_per_face = [None] * num_triangles
        indices = [None] * num_triangles
        j = 0
        for idx in triangle_per_face[i]:
            triangles_per_face[j] = triangles[idx]
            indices[j] = idx
            j += 1

        if num_triangles == 0:
            continue
            bvhs[i - start] = BVH()  # Default constructor
        else:
            bvhs[i - start] = BVH(triangles_per_face, indices)
    
    return bvhs


def perform_intersection_check(bvhs: List[BVH], num_bvhs: int, triangles: List[Triangle],
                               vertex_tri_centroids: List[UVFloat3], assign_indices_ptr: List[int],
                               num_indices: int, offset: int, triangle_per_face: List[Set[int]]):
    unique_intersections = []

    for i in range(num_indices):
        if assign_indices_ptr[i] < offset:
            continue

        cur_tri = triangles[i]
        cur_bvh = bvhs[assign_indices_ptr[i] - offset]

        if cur_bvh.nodes is None or len(cur_bvh.nodes) == 0:
            continue

        intersections = cur_bvh.intersect(cur_tri)

        if intersections:
            for intersect in intersections:
                if i != intersect:
                    if i < intersect:
                        unique_intersections.append((i, intersect))
                    else:
                        unique_intersections.append((intersect, i))

    # Step 2: Process unique intersections
    for idx in range(len(unique_intersections)):
        first, second = unique_intersections[idx]

        i_idx = assign_indices_ptr[first]
        norm_idx = i_idx % 6
        axis = 0 if norm_idx < 2 else (1 if norm_idx < 4 else 2)
        use_max = (i_idx % 2) == 1

        pos_a = vertex_tri_centroids[first].x if axis == 0 else (vertex_tri_centroids[first].y if axis == 1 else vertex_tri_centroids[first].z)
        pos_b = vertex_tri_centroids[second].x if axis == 0 else (vertex_tri_centroids[second].y if axis == 1 else vertex_tri_centroids[second].z)

        if use_max:
            if pos_a < pos_b:
                first, second = second, first
        else:
            if pos_a > pos_b:
                first, second = second, first

        unique_intersections[idx] = (first, second)

    # Now only get the second intersections from the pair and put them in a set
    second_intersections = set()
    for idx in range(len(unique_intersections)):
        second_intersections.add(unique_intersections[idx][1])

    for int_idx in second_intersections:
        intersect_idx = assign_indices_ptr[int_idx]
        new_index = intersect_idx + 6
        new_index = max(0, min(new_index, 12))

        assign_indices_ptr[int_idx] = new_index
        triangle_per_face[intersect_idx].remove(int_idx)
        triangle_per_face[new_index].add(int_idx)
    
    return assign_indices_ptr


def assign_faces_uv_to_atlas_index(vertices: np.ndarray, indices: np.ndarray, face_uv: np.ndarray,
                                    face_index: np.ndarray) -> np.ndarray:
    # Get the number of faces
    num_faces = indices.shape[0]
    assign_indices = np.empty(num_faces, dtype=np.int64)
    assign_indices[:] = face_index

    # Initialize arrays for the triangles and centroids
    vertex_tri_centroids = [None] * num_faces
    triangles = [None] * num_faces

    # Use a list of sets to store triangles for each face
    triangle_per_face = [set() for _ in range(13)]

    # Step 1: Build the triangle data and centroids
    for i in range(num_faces):
        face_idx = i * 3
        triangles[i] = Triangle(
            UVFloat3(face_uv[face_idx][0], face_uv[face_idx][1], 0),
            UVFloat3(face_uv[face_idx + 1][0], face_uv[face_idx + 1][1], 0),
            UVFloat3(face_uv[face_idx + 2][0], face_uv[face_idx + 2][1], 0)
        )
        vertex_tri_centroids[i] = triangles[i].centroid

        # Assign the triangle to the face index
        triangle_per_face[face_index[i]].add(i)

    # Step 2: Create BVHs for the first set
    bvhs = [None] * 6
    bvhs = create_bvhs(bvhs, triangles, triangle_per_face, num_faces, 0, 6)

    assign_indices = perform_intersection_check(bvhs, 6, triangles, vertex_tri_centroids, assign_indices, num_faces, 0, triangle_per_face)

    # Step 3: Create new BVHs for the second set
    new_bvhs = [None] * 6
    bvhs = create_bvhs(new_bvhs, triangles, triangle_per_face, num_faces, 6, 12)

    assign_indices = perform_intersection_check(new_bvhs, 6, triangles, vertex_tri_centroids, assign_indices, num_faces, 6, triangle_per_face)

    return assign_indices