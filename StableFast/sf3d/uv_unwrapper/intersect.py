from typing import List
from .common import Matrix4,UVFloat2, dot_3d, orient2d, EPSILON, UVFloat3, cross_3d, normalize, distance_3d, triangle_centroid_3d, orient3d
import numpy as np


class Triangle:
    def __init__(self, v0, v1, v2):
        if (type(v0) == UVFloat2):
            self.a = UVFloat3(v0.x, v0.y, 0)
        else:
            self.a = v0
        if (type(v1) == UVFloat2):
            self.b = UVFloat3(v1.x, v1.y, 0)
        else:
            self.b = v1
        if (type(v2) == UVFloat2):
            self.c = UVFloat3(v2.x, v2.y, 0)
        else:
            self.c = v2
        self.centroid = triangle_centroid_3d(self.a, self.b, self.c)

    def overlaps(self, other: 'Triangle') -> bool:
        return triangle_triangle_intersection(self.a, self.b, self.c, other.a, other.b, other.c)

    def get_normal(self) -> UVFloat3:
        u = self.b - self.a
        v = self.c - self.a
        return normalize(cross_3d(u, v))

    def is_degenerate(self) -> bool:
        u = self.a - self.b
        v = self.a - self.c
        cr = cross_3d(u, v)
        return abs(cr.x) < EPSILON and abs(cr.y) < EPSILON and abs(cr.z) < EPSILON
    
def triangle_aabb_intersection(
    aabb_min: UVFloat2, aabb_max: UVFloat2,
    v0: UVFloat2, v1: UVFloat2, v2: UVFloat2
) -> bool:
    """
    Check if a triangle intersects an Axis-Aligned Bounding Box (AABB).
    """
    # Extract AABB bounds
    left, right = aabb_min.x, aabb_max.x
    top, bottom = aabb_min.y, aabb_max.y

    # Test if any triangle vertex is inside the AABB
    for v in [v0, v1, v2]:
        if left <= v.x <= right and top <= v.y <= bottom:
            return True

    # Edge intersection test
    def edge_intersects(edge_start, edge_end):
        dx, dy = edge_end.x - edge_start.x, edge_end.y - edge_start.y
        if abs(dx) > EPSILON:
            # Vertical intersections
            t1 = (left - edge_start.x) / dx
            t2 = (right - edge_start.x) / dx
            y1 = edge_start.y + t1 * dy
            y2 = edge_start.y + t2 * dy
            if (top <= y1 <= bottom or top <= y2 <= bottom) and (0 <= t1 <= 1 or 0 <= t2 <= 1):
                return True

        if abs(dy) > EPSILON:
            # Horizontal intersections
            t1 = (top - edge_start.y) / dy
            t2 = (bottom - edge_start.y) / dy
            x1 = edge_start.x + t1 * dx
            x2 = edge_start.x + t2 * dx
            if (left <= x1 <= right or left <= x2 <= right) and (0 <= t1 <= 1 or 0 <= t2 <= 1):
                return True

        return False

    # Check if any edge intersects the AABB
    edges = [(v0, v1), (v1, v2), (v2, v0)]
    if any(edge_intersects(edge_start, edge_end) for edge_start, edge_end in edges):
        return True

    # Check if any AABB corner is inside the triangle
    aabb_corners = [
        aabb_min,
        UVFloat2(aabb_min.x, aabb_max.y),
        UVFloat2(aabb_max.x, aabb_min.y),
        aabb_max,
    ]
    for corner in aabb_corners:
        if orient2d(v0, v1, corner) >= 0 and orient2d(v1, v2, corner) >= 0 and orient2d(v2, v0, corner) >= 0:
            return True

    return False


def clip_triangle(t1: Triangle, t2: Triangle, intersections: List) -> List[UVFloat2]:
    """
    Clip one triangle (t2) against another triangle (t1).
    """
    clip = [t1.a, t1.b, t1.c]
    output = [t2.a, t2.b, t2.c]
    orients = [0] * len(output) * 3

    for i in range(3):  # Clip against each edge of t1
        input_polygon = output[:]
        output = []
        i_prev = (i + 2) % 3
        
        for j in range(len(input_polygon)):
            orients[j] = orient2d(clip[i_prev], clip[i], input_polygon[j])

        for j in range(len(input_polygon)):
            j_prev = (j - 1 + len(input_polygon)) % len(input_polygon)

            if orients[j] >= 0:
                if orients[j_prev] < 0:
                    inter = line_intersection(clip[i_prev], clip[i], input_polygon[j_prev], input_polygon[j])
                    output.append(UVFloat3(inter.x, inter.y, inter.z))
                output.append(UVFloat3(input_polygon[j].x, input_polygon[j].y, input_polygon[j].z))
            elif orients[j_prev] >= 0:
                inter = line_intersection(clip[i_prev], clip[i], input_polygon[j_prev], input_polygon[j])
                output.append(UVFloat3(inter.x, inter.y, inter.z))

        for vert in output:
            j = 0
            same_found = False
            while j < len(intersections) and not same_found:
                same_found = distance_3d(vert, intersections[j]) < 1e-6
                j += 1

            if not same_found:
                intersections.append(vert)

    return intersections


def line_intersection(a1: UVFloat2, b1: UVFloat2, a2: UVFloat2, b2: UVFloat2) -> UVFloat2:
    """
    Find the intersection point of two line segments.
    """
    dx1, dy1 = a1.x - b1.x, a1.y - b1.y
    dx2, dy2 = a2.x - b2.x, a2.y - b2.y

    denominator = dx1 * dy2 - dx2 * dy1
    
    n1 = a1.x * b1.y - a1.y * b1.x
    n2 = a2.x * b2.y - a2.y * b2.x

    return UVFloat3((n1 * dx2 - n2 * dx1) / denominator, (n1 * dy2 - n2 * dy1) / denominator, 0)


def make_tri_vertex_alone(tri: Triangle, oa: int, ob: int, oc: int):
    """
    Permute the vertices of the triangle so that one vertex is alone on one side.
    Args:
        tri: The triangle to modify.
        oa, ob, oc: Orientations of the vertices.
    """
    if oa == ob:
        # Vertex `c` is alone, permute right so `c` becomes `a`
        tri = permute_tri_right(tri)
    elif oa == oc:
        # Vertex `b` is alone, permute left so `b` becomes `a`
        tri = permute_tri_left(tri)
    elif ob != oc:
        # Ensure `a` is on the positive side
        if ob > 0:
            tri =permute_tri_left(tri)
        elif oc > 0:
            tri = permute_tri_right(tri)

    return tri


def make_tri_counter_clockwise(tri: Triangle):
    if (orient2d(tri.a, tri.b, tri.c) < 0):
        temp = tri.c
        tri.c = tri.b
        tri.b = temp

    return tri


def permute_tri_left(tri: Triangle):
    """
    Permute triangle vertices to the left (cyclic permutation).
    """
    temp = tri.a
    tri.a = tri.b
    tri.b = tri.c
    tri.c = temp
    return tri


def permute_tri_right(tri: Triangle):
    """
    Permute triangle vertices to the right (cyclic permutation).
    """
    temp = tri.c
    tri.c = tri.b
    tri.b = tri.a
    tri.a = temp
    return tri


def polygon_area(polygon: List[UVFloat3]) -> float:
    """
    Calculate the area of a polygon defined by a list of vertices.
    Args:
        polygon: A list of 3D vertices defining the polygon.
    Returns:
        The area of the polygon.
    """
    if len(polygon) < 3:
        return 0.0  # Not a valid polygon

    normal = UVFloat3(0.0, 0.0, 0.0)
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]  # Next vertex, wrapping around
        normal += cross_3d(p1, p2)

    return normal.magnitude() / 2.0  # Area is half the normal magnitude


def triangle_triangle_intersection(
    p1: UVFloat2, q1: UVFloat2, r1: UVFloat2,
    p2: UVFloat2, q2: UVFloat2, r2: UVFloat2
) -> bool:
    """
    Check if two triangles intersect and compute intersection area if applicable.
    """
    # Convert UVFloat2 to Triangle objects
    t1 = Triangle(p1, q1, r1)
    t2 = Triangle(p2, q2, r2)

    # Check for degeneracy
    if t1.is_degenerate() or t2.is_degenerate():
        return False

    # Determine 3D orientations for vertices of t1 relative to t2
    o1a = orient3d(t2.a, t2.b, t2.c, t1.a)
    o1b = orient3d(t2.a, t2.b, t2.c, t1.b)
    o1c = orient3d(t2.a, t2.b, t2.c, t1.c)

    # Perform cross intersection
    intersections = []
    if o1a == o1b and o1a == o1c:
        if o1a == 0:
            intersects, intersections = coplanar_intersect(t1, t2, intersections)
        else:
            intersects = False
    else:
        intersects, intersections = cross_intersect(t1, t2, o1a, o1b, o1c, intersections)

    if intersects:
        # Compute area of intersection polygon
        area = polygon_area(intersections)
        if area < 1e-10 or not np.isfinite(area):
            return False

    return intersects


def intersection_type_r1(t1, t2):
    p1 = t1.a
    q1 = t1.b
    r1 = t1.c
    p2 = t2.a
    r2 = t2.c

    # I
    if orient2d(r2, p2, q1) >= 0:
        # II.a
        if orient2d(r2, p1, q1) >= 0:
            # III.a
            if orient2d(p1, p2, q1) >= 0:
                return True
            else:
                # IV.a
                if orient2d(p1, p2, r1) >= 0:
                    # V
                    if orient2d(q1, r1, p2) >= 0:
                        return True
        else:
            # II.b
            if orient2d(r2, p2, r1) >= 0:
                # III.b
                if orient2d(q1, r1, r2) >= 0:
                    # IV.b
                    if orient2d(p1, p2, r1) >= 0:
                        return True
    return False


def intersection_type_r2(t1, t2):
    p1 = t1.a
    q1 = t1.b
    r1 = t1.c
    p2 = t2.a
    q2 = t2.b
    r2 = t2.c

    # I
    if orient2d(r2, p2, q1) >= 0:
        # II.a
        if orient2d(q2, r2, q1) >= 0:
            # III.a
            if orient2d(p1, p2, q1) >= 0:
                # IV.a
                if orient2d(p1, q2, q1) <= 0:
                    return True
            else:
                # IV.b
                if orient2d(p1, p2, r1) >= 0:
                    # V.a
                    if orient2d(r2, p2, r1) <= 0:
                        return True
        else:
            # III.b
            if orient2d(p1, q2, q1) <= 0:
                # IV.c
                if orient2d(q2, r2, r1) >= 0:
                    # V.b
                    if orient2d(q1, r1, q2) >= 0:
                        return True
    else:
        # II.b
        if orient2d(r2, p2, r1) >= 0:
            # III.c
            if orient2d(q1, r1, r2) >= 0:
                # IV.d
                if orient2d(r1, p1, p2) >= 0:
                    return True
            else:
                # IV.e
                if orient2d(q1, r1, q2) >= 0:
                    # V.c
                    if orient2d(q2, r2, r1) >= 0:
                        return True
    return False

def coplanar_intersect(t1: Triangle, t2: Triangle, intersections: List[UVFloat2]):
    """
    Handle coplanar intersection of two triangles.
    """
    normal = t1.get_normal()
    normal = normalize(normal)
    u = normalize(t1.a - t1.b)
    v = cross_3d(normal, u)

    u += t1.a
    v += t1.a
    normal += t1.a

    _matrix = Matrix4()
    _matrix.set(t1.a.x, u.x, v.x, normal.x, t1.a.y, u.y, v.y, normal.y, t1.a.z, u.z, v.z, normal.z, 1, 1, 1, 1)

    _affineMatrix = Matrix4()
    _affineMatrix.set(0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1)

    _matrix.invert()  # Invert the _matrix
    _matrix = _affineMatrix * _matrix

    t1.a = _matrix.apply_to_point(t1.a)
    t1.b = _matrix.apply_to_point(t1.b)
    t1.c = _matrix.apply_to_point(t1.c)
    t2.a = _matrix.apply_to_point(t2.a)
    t2.b = _matrix.apply_to_point(t2.b)
    t2.c = _matrix.apply_to_point(t2.c)

    t1 = make_tri_counter_clockwise(t1)
    t2 = make_tri_counter_clockwise(t2)

    p1 = t1.a
    p2 = t2.a
    q2 = t2.b
    r2 = t2.c

    o_p2q2 = orient2d(p2, q2, p1)
    o_q2r2 = orient2d(q2, r2, p1)
    o_r2p2 = orient2d(r2, p2, p1)

    intersecting = False
    if o_p2q2 >= 0:
        if o_q2r2 >= 0:
            if o_r2p2 >= 0:
                intersecting = True  # + + +
            else:
                intersecting = intersection_type_r1(t1, t2)  # + + -
        else:
            if o_r2p2 >= 0:
                t2 = permute_tri_right(t2)
                intersecting = intersection_type_r1(t1, t2)  # + - +
            else:
                intersecting = intersection_type_r2(t1, t2)  # + - -
    else:
        if o_q2r2 >= 0:
            if o_r2p2 >= 0:
                t2 = permute_tri_left(t2)
                intersecting = intersection_type_r1(t1, t2)  # - + +
            else:
                t2 = permute_tri_left(t2)
                intersecting = intersection_type_r2(t1, t2)  # - + -
        else:
            if o_r2p2 >= 0:
                t2 = permute_tri_right(t2)
                intersecting = intersection_type_r2(t1, t2)  # - - +
            else:
                print("Triangles should not be flat.")
                return False, []

    if intersecting:
        intersections = clip_triangle(t1, t2, intersections)

        _matrix.invert()
        for point in intersections:
            point = _matrix.apply_to_point(point)

    return intersecting, intersections

def make_tri_vertex_positive(tri: Triangle, other: Triangle):
    """
    Adjust the triangle's orientation so that it lies on the positive side of another triangle.
    This ensures the correct orientation for further intersection calculations.
    
    Args:
        tri: The triangle to adjust.
        other: The triangle that we want to compare against.
    """
    # Compute the orientation of tri relative to the other triangle
    o = orient3d(other.a, other.b, other.c, tri.a)
    
    # If the orientation is negative, permute the vertices of tri to make it positive
    if o < 0:
        tri.b, tri.c = tri.c, tri.b

    return tri

def intersectPlane(a: UVFloat3, b: UVFloat3, p: UVFloat3, n: UVFloat3, target: UVFloat3):
    """
    Intersect a line (a, b) with a plane defined by point p and normal vector n.
    
    Args:
        a: The start point of the line.
        b: The end point of the line.
        p: A point on the plane.
        n: The normal vector of the plane.
        target: The intersection point of the line and the plane.
    """
    # Vector from a to b (direction of the line)
    u = b - a
    v = a - p
    
    # Dot products
    dot1 = dot_3d(n, u)
    dot2 = dot_3d(n, v)
    
    # Compute the scalar factor t for the intersection point
    u = u * (-dot2 / dot1)
    
    # Find the intersection point
    target.x = a.x + u.x
    target.y = a.y + u.y
    target.z = a.z + u.z

    return target

def compute_line_intersection(t1: Triangle, t2: Triangle, target: List[UVFloat3]):
    """
    Compute intersection points between two triangles and store them in target list.
    
    Args:
        t1: The first triangle.
        t2: The second triangle.
        target: A list to store intersection points.
    """
    # Get the normals of the two triangles
    n1, n2 = t1.get_normal(n1), t2.get_normal(n2)

    # Check orientation of the triangle vertices relative to each other
    o1 = orient3d(t1.a, t1.c, t2.b, t2.a)  # Orientation of t1 relative to t2
    o2 = orient3d(t1.a, t1.b, t2.c, t2.a)  # Orientation of t1 relative to t2

    # Compute the intersection points based on the orientations
    i1, i2 = UVFloat3(), UVFloat3()  # Intersection points

    if o1 > 0:
        if o2 > 0:
            # Case 1: Both orientations positive
            i1 = intersectPlane(t1.a, t1.c, t2.a, n2, i1)
            i2 = intersectPlane(t2.a, t2.c, t1.a, n1, i2)
        else:
            # Case 2: One orientation positive, other negative
            i1 = intersectPlane(t1.a, t1.c, t2.a, n2, i1)
            i2 = intersectPlane(t1.a, t1.b, t2.a, n2, i2)
    else:
        if o2 > 0:
            # Case 3: One orientation negative, other positive
            i1 = intersectPlane(t2.a, t2.b, t1.a, n1, i1)
            i2 = intersectPlane(t2.a, t2.c, t1.a, n1, i2)
        else:
            # Case 4: Both orientations negative
            i1 = intersectPlane(t2.a, t2.b, t1.a, n1, i1)
            i2 = intersectPlane(t1.a, t1.b, t2.a, n2, i2)

    # Add the intersection points to the target list
    target.append(i1)
    if distance_3d(i1, i2) >= EPSILON:
        target.append(i2)

    return target


def cross_intersect(
    t1: Triangle, t2: Triangle, o1a: int, o1b: int, o1c: int, intersections: List[UVFloat2]
) -> bool:
    """
    Handle cross intersection of two triangles.
    """
    o2a = orient3d(t1.a, t1.b, t1.c, t2.a)
    o2b = orient3d(t1.a, t1.b, t1.c, t2.b)
    o2c = orient3d(t1.a, t1.b, t1.c, t2.c)

    if o2a == o2b and o2a == o2c:
        return False

    # Orient t1 and t2 for intersection computation
    t1 = make_tri_vertex_alone(t1, o1a, o1b, o1c)
    t2 = make_tri_vertex_alone(t2, o2a, o2b, o2c)

    # Ensure the vertex on the positive side
    t2 = make_tri_vertex_positive(t2, t1)
    t1 = make_tri_vertex_positive(t1, t2)

    # Perform oriented intersection checks
    o1 = orient3d(t1.a, t1.b, t2.a, t2.b)
    o2 = orient3d(t1.a, t1.c, t2.c, t2.a)

    if o1 <= 0 and o2 <= 0:
        intersections = compute_line_intersection(t1, t2, intersections)
        return True, intersections
    return False, intersections
