import numpy as np
import time
from scipy.spatial import Delaunay
from math import cos, sin, acos, sqrt


class SymetricMatrix:
    def __init__(self, *args):
        """
        Constructor for SymetricMatrix.
        
        - If no arguments are provided, initializes the matrix with zero.
        - If one argument is provided, initializes all elements with that value.
        - If ten arguments are provided, initializes with those values.
        - If four arguments are provided, creates a matrix using the plane equation.
        """
        if len(args) == 0:
            self.m = [0.0] * 10
        elif len(args) == 1:
            self.m = [args[0]] * 10
        elif len(args) == 10:
            self.m = list(args)
        elif len(args) == 4:
            a, b, c, d = args
            self.m = [
                a * a, a * b, a * c, a * d,
                b * b, b * c, b * d,
                c * c, c * d,
                d * d
            ]
        else:
            raise ValueError("Invalid number of arguments for constructor")

    def __getitem__(self, c):
        """
        Get item at index c.
        """
        return self.m[c]

    def det(self, a11, a12, a13, a21, a22, a23, a31, a32, a33):
        """
        Calculate the determinant of the matrix.
        """
        det = (
            self.m[a11] * self.m[a22] * self.m[a33] +
            self.m[a13] * self.m[a21] * self.m[a32] +
            self.m[a12] * self.m[a23] * self.m[a31] -
            self.m[a13] * self.m[a22] * self.m[a31] -
            self.m[a11] * self.m[a23] * self.m[a32] -
            self.m[a12] * self.m[a21] * self.m[a33]
        )
        return det

    def __add__(self, other):
        result = SymetricMatrix()
        for i in range(10):
            result.m[i] = self.m[i] + other.m[i]
        return result

    def __iadd__(self, other):
        """
        In-place addition operator overload.
        """
        for i in range(10):
            self.m[i] += other[i]
        return self

    def __repr__(self):
        """
        String representation of the matrix.
        """
        return f"SymetricMatrix({', '.join(map(str, self.m))})"


class vec3f:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def normalize(self):
        norm = np.linalg.norm([self.x, self.y, self.z])
        if norm == 0:
            return self
        self.x /= norm
        self.y /= norm
        self.z /= norm
        return self

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, v1, v2):
        cross_product = np.cross([v1.x, v1.y, v1.z], [v2.x, v2.y, v2.z])
        self.x, self.y, self.z = cross_product
        return self

    def __sub__(self, other):
        return vec3f(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __add__(self, other):
        return vec3f(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def __mul__(self, other):
        if isinstance(other, vec3f):
            return vec3f(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            return vec3f(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        if isinstance(other, vec3f):
            return vec3f(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
            return vec3f(self.x / other, self.y / other, self.z / other)

    def cross(self, a, b):
        self.x = a.y * b.z - a.z * b.y
        self.y = a.z * b.x - a.x * b.z
        self.z = a.x * b.y - a.y * b.x
        return self

    def length(self):
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self, desired_length=1):
        square = self.length()
        if square <= 0.00001:
            self.x = 1
            self.y = 0
            self.z = 0
            return self
        scale = desired_length / square
        self.x *= scale
        self.y *= scale
        self.z *= scale
        return self

    def angle(self, v):
        dot = self.dot(v)
        len_product = self.length() * v.length()
        if len_product == 0:
            len_product = 0.00001
        input_val = dot / len_product
        input_val = max(min(input_val, 1), -1)
        return acos(input_val)

    def angle2(self, v, w):
        dot = self.dot(v)
        len_product = self.length() * v.length()
        if len_product == 0:
            len_product = 1
        plane = vec3f().cross(self, w)
        if plane.dot(v) > 0:
            return -acos(dot / len_product)
        return acos(dot / len_product)

    def rot_x(self, a):
        yy = cos(a) * self.y + sin(a) * self.z
        zz = cos(a) * self.z - sin(a) * self.y
        self.y = yy
        self.z = zz
        return self

    def rot_y(self, a):
        xx = cos(-a) * self.x + sin(-a) * self.z
        zz = cos(-a) * self.z - sin(-a) * self.x
        self.x = xx
        self.z = zz
        return self

    def rot_z(self, a):
        yy = cos(a) * self.y + sin(a) * self.x
        xx = cos(a) * self.x - sin(a) * self.y
        self.y = yy
        self.x = xx
        return self

    def clamp(self, min_val, max_val):
        self.x = max(min(self.x, max_val), min_val)
        self.y = max(min(self.y, max_val), min_val)
        self.z = max(min(self.z, max_val), min_val)

    def invert(self):
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z
        return self

    def frac(self):
        return vec3f(self.x - int(self.x), self.y - int(self.y), self.z - int(self.z))

    def integer(self):
        return vec3f(int(self.x), int(self.y), int(self.z))

    def __repr__(self):
        return f"vec3f({self.x}, {self.y}, {self.z})"

class Triangle:
    def __init__(self, v=[0, 0, 0], err=[0.0, 0.0, 0.0, 0.0], deleted=0, dirty=0, n=vec3f()):
        self.v = v
        self.err = err
        self.deleted = deleted
        self.dirty = dirty
        self.n = n

class Vertex:
    def __init__(self, p=vec3f(), tstart=0, triangleCount=0, q=SymetricMatrix(), border=0):
        self.p = p
        self.tstart = tstart
        self.triangleCount = triangleCount
        self.q = q
        self.border = border

class Ref:
    def __init__(self, tid=0, tvertex=0):
        self.tid = tid
        self.tvertex = tvertex

# Global variables
triangles = []
vertices = []
refs = []

def simplify_mesh(target_count, aggressiveness=7):
    time_start = time.time()

    for triangle in triangles:
        triangle.deleted = 0

    deleted_triangles = 0
    deleted0 = []
    deleted1 = []
    triangle_count = len(triangles)

    for iteration in range(100):
        print(f"iteration {iteration} - triangles {triangle_count - deleted_triangles}", end='\r')
        if triangle_count - deleted_triangles <= target_count:
            break

        if iteration % 5 == 0:
            update_mesh(iteration)

        for triangle in triangles:
            triangle.dirty = 0

        threshold = 0.000000001 * pow(float(iteration + 3), aggressiveness)

        for triangle in triangles:
            if triangle.err[3] > threshold or triangle.deleted or triangle.dirty:
                continue

            for j in range(3):
                if triangle.err[j] < threshold:
                    vertIdx1 = triangle.v[j]
                    vert1 = vertices[vertIdx1]
                    vertIdx2 = triangle.v[(j + 1) % 3]
                    vert2 = vertices[vertIdx2]

                    if vert1.border != vert2.border:
                        continue

                    p = vec3f()
                    calculate_error(vertIdx1, vertIdx2, p)

                    deleted0 = [0] * vert1.triangleCount
                    deleted1 = [0] * vert2.triangleCount

                    if flipped(p, vertIdx1, vertIdx2, vert1, vert2, deleted0):
                        continue
                    if flipped(p, vertIdx2, vertIdx1, vert2, vert1, deleted1):
                        continue

                    vert1.p = p
                    vert1.q += vert2.q
                    tstart = len(refs)

                    deleted_triangles += update_triangles(vertIdx1, vert1, deleted0)
                    deleted_triangles += update_triangles(vertIdx1, vert2, deleted1)

                    triangleCount = len(refs) - tstart

                    if triangleCount <= vert1.triangleCount:
                        refs[vert1.tstart:vert1.tstart + triangleCount] = refs[tstart:tstart + triangleCount]
                    else:
                        vert1.tstart = tstart

                    vert1.triangleCount = triangleCount
                    break

            if triangle_count - deleted_triangles <= target_count:
                break

    compact_mesh()

    time_end = time.time()
    print(f"simplify_mesh - {triangle_count - deleted_triangles}/{triangle_count} {deleted_triangles * 100 / triangle_count:.2f}% removed in {time_end - time_start:.2f} seconds")


def flipped(p, i0, i1, v0, v1, deleted):
    bordercount = 0
    for k in range(v0.triangleCount):
        t = triangles[refs[v0.tstart + k].tid]
        if t.deleted:
            continue

        s = refs[v0.tstart + k].tvertex
        id1 = t.v[(s + 1) % 3]
        id2 = t.v[(s + 2) % 3]

        if id1 == i1 or id2 == i1:
            bordercount += 1
            deleted[k] = 1
            continue

        d1 = vertices[id1].p - p
        d1.normalize()
        d2 = vertices[id2].p - p
        d2.normalize()

        if abs(d1.dot(d2)) > 0.999:
            return True

        n = vec3f()
        n.cross(d1, d2)
        n.normalize()
        deleted[k] = 0

        if n.dot(t.n) < 0.2:
            return True

    return False


def update_triangles(i0, v, deleted):
    deletedCount = 0
    p = vec3f()
    for k in range(v.triangleCount):
        r = refs[v.tstart + k]
        t = triangles[r.tid]
        if t.deleted:
            continue
        if deleted[k]:
            t.deleted = 1
            deletedCount += 1
            continue

        t.v[r.tvertex] = i0
        t.dirty = 1
        t.err[0] = calculate_error(t.v[0], t.v[1], p)
        t.err[1] = calculate_error(t.v[1], t.v[2], p)
        t.err[2] = calculate_error(t.v[2], t.v[0], p)
        t.err[3] = min(t.err[0], min(t.err[1], t.err[2]))
        refs.append(r)
    return deletedCount

    
def update_mesh(iteration):
    if iteration > 0:  # compact triangles
        dst = 0
        for i in range(len(triangles)):
            if not triangles[i].deleted:
                triangles[dst] = triangles[i]
                dst += 1
        del triangles[dst:]

    # Init Reference ID list
    for i in range(len(vertices)):
        vertices[i].tstart = 0
        vertices[i].triangleCount = 0
    for i in range(len(triangles)):
        t = triangles[i]
        for j in range(3):
            vertices[t.v[j]].triangleCount += 1

    tstart = 0
    for i in range(len(vertices)):
        v = vertices[i]
        v.tstart = tstart
        tstart += v.triangleCount
        v.triangleCount = 0

    # Write References
    global refs
    refs = [Ref() for _ in range(len(triangles) * 3)]
    for i in range(len(triangles)):
        t = triangles[i]
        for j in range(3):
            v = vertices[t.v[j]]
            refs[v.tstart + v.triangleCount].tid = i
            refs[v.tstart + v.triangleCount].tvertex = j
            v.triangleCount += 1
    
    if iteration == 0:
        # Identify boundary : vertices[].border = 0,1
        vcount = []
        vids = []

        for i in range(len(vertices)):
            vertices[i].border = 0

        for i in range(len(vertices)):
            v = vertices[i]
            vcount.clear()
            vids.clear()
            for j in range(v.triangleCount):
                k = refs[v.tstart + j].tid
                t = triangles[k]
                for k in range(3):
                    ofs = 0
                    id = t.v[k]
                    while ofs < len(vcount):
                        if vids[ofs] == id:
                            break
                        ofs += 1
                    if ofs == len(vcount):
                        vcount.append(1)
                        vids.append(id)
                    else:
                        vcount[ofs] += 1
            for j in range(len(vcount)):
                if vcount[j] == 1:
                    vertices[vids[j]].border = 1

        # Initialize errors
        for i in range(len(vertices)):
            vertices[i].q = SymetricMatrix(0.0)

        for i in range(len(triangles)):
            t = triangles[i]
            p = [vec3f() for _ in range(3)]
            for j in range(3):
                p[j] = vertices[t.v[j]].p
            n = vec3f()
            n.cross(p[1] - p[0], p[2] - p[0])
            n.normalize()
            t.n = n
            for j in range(3):
                vertices[t.v[j]].q = vertices[t.v[j]].q + SymetricMatrix(n.x, n.y, n.z, -n.dot(p[0]))

        for i in range(len(triangles)):
            # Calc Edge Error
            t = triangles[i]
            p = vec3f()
            for j in range(3):
                t.err[j] = calculate_error(t.v[j], t.v[(j + 1) % 3], p)
            t.err[3] = min(t.err[0], min(t.err[1], t.err[2]))


def compact_mesh():
    dst = 0
    for i in range(len(vertices)):
        vertices[i].triangleCount = 0

    for i in range(len(triangles)):
        if not triangles[i].deleted:
            t = triangles[i]
            triangles[dst] = t
            dst += 1
            for j in range(3):
                vertices[t.v[j]].triangleCount = 1

    del triangles[dst:]

    dst = 0
    for i in range(len(vertices)):
        if vertices[i].triangleCount:
            vertices[i].tstart = dst
            vertices[dst].p = vertices[i].p
            dst += 1

    for i in range(len(triangles)):
        t = triangles[i]
        for j in range(3):
            t.v[j] = vertices[t.v[j]].tstart

    del vertices[dst:]


def vertex_error(q, x, y, z):
    return (q[0] * x**2 + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x + q[4] * y**2
            + 2 * q[5] * y * z + 2 * q[6] * y + q[7] * z**2 + 2 * q[8] * z + q[9])


def calculate_error(id_v1, id_v2, p_result):
    q = vertices[id_v1].q + vertices[id_v2].q
    border = vertices[id_v1].border & vertices[id_v2].border
    error = 0
    det = q.det(0, 1, 2, 1, 4, 5, 2, 5, 7)

    if det != 0 and not border:
        p_result.x = -1 / det * q.det(1, 2, 3, 4, 5, 6, 5, 7, 8)
        p_result.y = 1 / det * q.det(0, 2, 3, 1, 5, 6, 2, 7, 8)
        p_result.z = -1 / det * q.det(0, 1, 3, 1, 4, 6, 2, 5, 8)
        error = vertex_error(q, p_result.x, p_result.y, p_result.z)
    else:
        p1 = vertices[id_v1].p
        p2 = vertices[id_v2].p
        p3 = (p1 + p2) / 2
        error1 = vertex_error(q, p1.x, p1.y, p1.z)
        error2 = vertex_error(q, p2.x, p2.y, p2.z)
        error3 = vertex_error(q, p3.x, p3.y, p3.z)
        error = min(error1, error2, error3)
        if error1 == error:
            p_result = p1
        elif error2 == error:
            p_result = p2
        else:
            p_result = p3

    return error


def write_obj(filename):
    try:
        with open(filename, 'w') as file:
            for vertex in vertices:
                file.write(f"v {vertex.p.x} {vertex.p.y} {vertex.p.z}\n")
            for triangle in triangles:
                if not triangle.deleted:
                    file.write(f"f {triangle.v[0] + 1} {triangle.v[1] + 1} {triangle.v[2] + 1}\n")
    except IOError:
        print(f"write_obj: can't write data file \"{filename}\".")
        exit(0)


def load_obj(filename, process_uv=False):
    global vertices, triangles, materials, mtllib
    vertices.clear()
    triangles.clear()
    uvs = []
    uv_map = []

    if not filename:
        return

    try:
        with open(filename, 'r') as fn:
            vertex_cnt = 0
            material = -1
            material_map = {}

            for line in fn:
                v = Vertex()
                uv = vec3f()

                if line.startswith("mtllib"):
                    mtllib = line[7:].strip()
                elif line.startswith("usemtl"):
                    usemtl = line[7:].strip()
                    if usemtl not in material_map:
                        material_map[usemtl] = len(materials)
                        materials.append(usemtl)
                    material = material_map[usemtl]
                elif line.startswith("vt "):
                    parts = list(map(float, line.split()[1:]))
                    if len(parts) == 2:
                        uv = vec3f(parts[0], parts[1], 0.0)
                    elif len(parts) == 3:
                        uv = vec3f(parts[0], parts[1], parts[2])
                    uvs.append(uv)
                elif line.startswith("v "):
                    parts = list(map(float, line.split()[1:]))
                    if len(parts) == 3:
                        v.p = vec3f(parts[0], parts[1], parts[2])
                        vertices.append(v)
                    elif len(parts) == 6:
                        v.p = vec3f(parts[0], parts[1], parts[2])
                        vertices.append(v)
                elif line.startswith("f "):
                    integers = list(map(int, line.replace('//', ' ').replace('/', ' ').split()[1:]))
                    t = Triangle()
                    tri_ok = False
                    has_uv = False

                    if len(integers) == 3:
                        tri_ok = True
                    elif len(integers) == 6 and integers[0] == integers[3] == integers[1] == integers[4] == integers[2] == integers[5]:
                        tri_ok = True
                    elif len(integers) == 9:
                        tri_ok = True
                        has_uv = True

                    if tri_ok:
                        t.v = [integers[0] - 1 - vertex_cnt, integers[1] - 1 - vertex_cnt, integers[2] - 1 - vertex_cnt]
                        t.attr = 0

                        if process_uv and has_uv:
                            indices = [integers[3] - 1 - vertex_cnt, integers[4] - 1 - vertex_cnt, integers[5] - 1 - vertex_cnt]
                            uv_map.append(indices)

                        t.material = material
                        triangles.append(t)
    except IOError:
        print(f"File {filename} not found!")
        return

    if process_uv and uvs:
        for i in range(len(triangles)):
            for j in range(3):
                triangles[i].uvs[j] = uvs[uv_map[i][j]]

    
load_obj(filename="C:/Users/shrav/Downloads/mesh.obj")

print(f"Input: {len(triangles)} triangles and {len(vertices)} vertices")
simplify_mesh(target_count=42000)
print(f"Output: {len(triangles)} triangles and {len(vertices)} vertices")


write_obj(filename="C:/Users/shrav/Downloads/mesh_simplified.obj")