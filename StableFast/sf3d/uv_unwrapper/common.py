import numpy as np

EPSILON = 1e-7


# Class to represent a 2D point or vector
class UVFloat2:
    def __init__(self, x=0.0, y=0.0):
        self.data = np.array([x, y], dtype=np.float32)

    @property
    def x(self):
        return self.data[0]

    @x.setter
    def x(self, value):
        self.data[0] = value

    @property
    def y(self):
        return self.data[1]

    @y.setter
    def y(self, value):
        self.data[1] = value

    def __getitem__(self, idx):
        if idx > 1:
            raise IndexError("bad index")
        return self.data[idx]

    def __setitem__(self, idx, value):
        if idx > 1:
            raise IndexError("bad index")
        self.data[idx] = value

    def __eq__(self, other):
        return np.allclose(self.data, other.data)

    def __sub__(self, other):
        return UVFloat2(*(self.data - other.data))

    def __add__(self, other):
        return UVFloat2(*(self.data + other.data))

    def __mul__(self, scalar):
        return UVFloat2(*(self.data * scalar))

    def magnitude(self):
        return np.sqrt(self.data[0]**2 + self.data[1]**2)

    def normalize(self):
        length = self.magnitude()
        if length > EPSILON:
            self.data /= length
        return self


# Class to represent a 3D point or vector
class UVFloat3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.data = np.array([x, y, z], dtype=np.float32)

    @property
    def x(self):
        return self.data[0]

    @x.setter
    def x(self, value):
        self.data[0] = value

    @property
    def y(self):
        return self.data[1]

    @y.setter
    def y(self, value):
        self.data[1] = value

    @property
    def z(self):
        return self.data[2]

    @z.setter
    def z(self, value):
        self.data[2] = value

    def __getitem__(self, idx):
        if idx > 2:
            raise IndexError("bad index")
        return self.data[idx]

    def __setitem__(self, idx, value):
        if idx > 2:
            raise IndexError("bad index")
        self.data[idx] = value

    def __eq__(self, other):
        return np.allclose(self.data, other.data)

    def __sub__(self, other):
        return UVFloat3(*(self.data - other.data))

    def __add__(self, other):
        return UVFloat3(*(self.data + other.data))

    def __mul__(self, scalar):
        return UVFloat3(*(self.data * scalar))

    def magnitude(self):
        return np.sqrt(np.sum(self.data**2))

    def normalize(self):
        length = self.magnitude()
        if length > EPSILON:
            self.data /= length
        return self


# Helper functions for vector operations
def calc_mean(a, b, c):
    return (a + b + c) / 3


def triangle_centroid_2d(v0: UVFloat2, v1: UVFloat2, v2: UVFloat2):
    return UVFloat2(
        calc_mean(v0.x, v1.x, v2.x),
        calc_mean(v0.y, v1.y, v2.y)
    )


def triangle_centroid_3d(v0: UVFloat3, v1: UVFloat3, v2: UVFloat3):
    return UVFloat3(
        calc_mean(v0.x, v1.x, v2.x),
        calc_mean(v0.y, v1.y, v2.y),
        calc_mean(v0.z, v1.z, v2.z)
    )


def dot_2d(a: UVFloat2, b: UVFloat2):
    return a.x * b.x + a.y * b.y


def dot_3d(a: UVFloat3, b: UVFloat3):
    return a.x * b.x + a.y * b.y + a.z * b.z


def cross_2d(a: UVFloat2, b: UVFloat2):
    return a.x * b.y - a.y * b.x


def cross_3d(a: UVFloat3, b: UVFloat3):
    return UVFloat3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    )


def normalize(v: UVFloat3) -> UVFloat3:
    length = np.sqrt(v.x**2 + v.y**2 + v.z**2)
    if length > EPSILON:
        return v * (1 / length)
    return v

def distance_2d(a: UVFloat2, b: UVFloat2):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


def distance_3d(a: UVFloat3, b: UVFloat3):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def orient2d(a: UVFloat2, b: UVFloat2, c: UVFloat2) -> int:
    det = a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)
    if det > EPSILON:
        return 1
    elif det < -EPSILON:
        return -1
    return 0


def orient3d(a: UVFloat2, b: UVFloat2, c: UVFloat2, d: UVFloat2) -> int:
    _matrix = Matrix4()
    _matrix.set(a.x, a.y, a.z, 1, b.x, b.y, b.z, 1, c.x, c.y, c.z, 1, d.x, d.y,
               d.z, 1)
    det = _matrix.determinant()
    if det > EPSILON:
        return 1
    elif det < -EPSILON:
        return -1
    return 0


# Class for 4x4 matrices
class Matrix4:
    def __init__(self):
        self.m = np.eye(4, dtype=np.float32)

    def set(self, *values):
        self.m = np.array(values, dtype=np.float32).reshape((4, 4))

    def determinant(self):
        return np.linalg.det(self.m)

    def __mul__(self, other):
        if isinstance(other, Matrix4):
            result = Matrix4()
            result.m = np.dot(self.m, other.m)
            return result
        elif isinstance(other, float) or isinstance(other, int):
            result = Matrix4()
            result.m = self.m * other
            return result
        else:
            raise TypeError("Unsupported multiplication")

    def __add__(self, other):
        result = Matrix4()
        result.m = self.m + other.m
        return result

    def __sub__(self, other):
        result = Matrix4()
        result.m = self.m - other.m
        return result

    def invert(self):
        try:
            self.m = np.linalg.inv(self.m)
            return True
        except np.linalg.LinAlgError:
            return False

    def apply_to_point(self, v: UVFloat3):
        vec = np.array([v.x, v.y, v.z, 1.0], dtype=np.float32)
        transformed = np.dot(self.m, vec)
        w = transformed[3]
        if abs(w) > EPSILON:
            transformed[:3] /= w
        return UVFloat3(*transformed[:3])

