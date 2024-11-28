import numpy as np
from typing import List, Set
from .bvh import BVH, Triangle
from .common import UVFloat3, UVFloat2, triangle_centroid_3d
import concurrent.futures


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
            bvhs[i - start] = BVH()  # Default constructor
        else:
            bvhs[i - start] = BVH(triangles_per_face, indices)
    
    return bvhs


def process_intersections(i, assign_indices_ptr, offset, triangles, bvhs, unique_intersections):
    if assign_indices_ptr[i] < offset:
        return []

    cur_tri = triangles[i]
    cur_bvh = bvhs[assign_indices_ptr[i] - offset]

    if cur_bvh.nodes is None or len(cur_bvh.nodes) == 0:
        return []

    intersections = cur_bvh.intersect(cur_tri)

    result = []
    if intersections:
        for intersect in intersections:
            if i != intersect:
                if i < intersect:
                    result.append((i, intersect))
                else:
                    result.append((intersect, i))
    return result

def parallel_process(num_indices, assign_indices_ptr, offset, triangles, bvhs):
    unique_intersections = []
    
    # Use ThreadPoolExecutor to parallelize the loop
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_intersections, i, assign_indices_ptr, offset, triangles, bvhs, unique_intersections)
            for i in range(num_indices)
        ]
        
        # Gather all results
        for future in concurrent.futures.as_completed(futures):
            intersections = future.result()
            unique_intersections.extend(intersections)

    return unique_intersections

def perform_intersection_check(bvhs: List[BVH], num_bvhs: int, triangles: List[Triangle],
                               vertex_tri_centroids: List[UVFloat3], assign_indices_ptr: List[int],
                               num_indices: int, offset: int, triangle_per_face: List[Set[int]]):
    
    unique_intersections = parallel_process(num_indices, assign_indices_ptr, offset, triangles, bvhs)

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
    
    return assign_indices_ptr, triangle_per_face


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
            UVFloat2(face_uv[face_idx][0], face_uv[face_idx][1]),
            UVFloat2(face_uv[face_idx + 1][0], face_uv[face_idx + 1][1]),
            UVFloat2(face_uv[face_idx + 2][0], face_uv[face_idx + 2][1])
        )

        v0 = UVFloat3(vertices[indices[i][0]][0], vertices[indices[i][0]][1], vertices[indices[i][0]][2])
        v1 = UVFloat3(vertices[indices[i][1]][0], vertices[indices[i][1]][1], vertices[indices[i][1]][2])
        v2= UVFloat3(vertices[indices[i][2]][0], vertices[indices[i][2]][1], vertices[indices[i][2]][2])

        vertex_tri_centroids[i] = triangle_centroid_3d(v0, v1, v2)

        # Assign the triangle to the face index
        triangle_per_face[face_index[i]].add(i)

    # Step 2: Create BVHs for the first set
    bvhs = [None] * 6
    
    import time
    t1 = time.time()
    bvhs = create_bvhs(bvhs, triangles, triangle_per_face, num_faces, 0, 6)
    
    t2 = time.time()
    print("BVH Create time:", t2 - t1)
    assign_indices, triangle_per_face = perform_intersection_check(bvhs, 6, triangles, vertex_tri_centroids, assign_indices, num_faces, 0, triangle_per_face)
    
    t3 = time.time()
    print("Intersection Check time:", t3 - t2)
    # Step 3: Create new BVHs for the second set
    new_bvhs = [None] * 6
    new_bvhs = create_bvhs(new_bvhs, triangles, triangle_per_face, num_faces, 6, 12)

    assign_indices, triangle_per_face = perform_intersection_check(new_bvhs, 6, triangles, vertex_tri_centroids, assign_indices, num_faces, 6, triangle_per_face)

    return assign_indices