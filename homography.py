"""Homography transformation matrices

The 3x3 homography transformation matrix for transformations in two
dimensions is illustrated below.

  | x0 |   | a  b  c | | x |
  | y0 | = | d  e  f | | y |
  | z0 |   | g  h  1 | | 1 |

x' = x0 / z0
y' = y0 / z0

"""
import affine
import numpy as np
import shapely.geometry

try:
    import cv2  # NOQA
    no_cv2 = False
except ImportError:
    no_cv2 = True


class Homography(object):
    def __init__(self, other=None):
        """
        Constructs itself from Homography, Affine or 3x3 mat
        default is the identity Homography.
        """
        if other is None:
            self.h = np.identity(3, dtype=np.float64)
        elif isinstance(other, Homography):
            self.h = other.h
        elif isinstance(other, affine.Affine):
            self.h = self._array_from_affine(other)
        else:
            self.h = np.asarray(other, dtype=np.float64)

        self.h /= self.h[2, 2]

    @classmethod
    def identity(cls):
        return cls()

    @classmethod
    def translation(cls, x, y):
        arr = np.array([[1, 0, x],
                        [0, 1, y],
                        [0, 0, 1]], dtype=np.float64)
        return cls(arr)

    @classmethod
    def scale(cls, x, y=None):
        if y is None:
            y = x
        arr = np.array([[x, 0, 0],
                        [0, y, 0],
                        [0, 0, 1]], dtype=np.float64)
        return cls(arr)

    @classmethod
    def rotation(cls, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        sin_ = np.sin(angle_rad)
        cos_ = np.cos(angle_rad)
        arr = np.array([[cos_, sin_, 0],
                        [-sin_, cos_, 0],
                        [0, 0, 1]], dtype=np.float64)
        return cls(arr)

    @classmethod
    def _array_from_affine(cls, aff):
        """
        :param aff: affine.Affine
        :return ndarray 3x3
        """
        a, b, c, d, e, f, _, _, _ = aff
        return np.array([[a, b, c],
                        [d, e, f],
                        [0, 0, 1]], dtype=np.float64)

    @classmethod
    def from_affine(cls, aff):
        """
        :param aff: affine.Affine
        :return Homography
        """
        arr = cls._array_from_affine(aff)
        return cls(arr)

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d['matrix']))

    def to_affine(self):
        """ Ignores projective part. """
        # TODO be smarter here, get closest affine
        mat = self.h / self.h[2, 2]
        return affine.Affine(mat[0, 0], mat[0, 1], mat[0, 2],
                             mat[1, 0], mat[1, 1], mat[1, 2])

    def to_dict(self):
        return {'matrix': self.h.tolist()}

    def projectivity(self):
        """ Rough approximation of how non-affine is the homography. """
        proj = np.max(np.abs(self.h[2, 0:2]))
        scale = np.sqrt(np.abs(np.linalg.det(self.h)))
        return proj / scale

    def apply(self, x, y, z=1):
        """
        Direct multiplication with H, returns 3-vector.
        deprecated? consider using __call__()
        """
        vec = np.array([x, y, z], dtype=np.float64)
        res = np.dot(self.h, vec)
        return res

    def __eq__(self, other):
        return self.equal(other)

    def equal(self, other, eps=1e-6):
        dist = self.dist(other, 1, 1)
        return dist < eps

    def norm(self, width=1, height=1):
        """
        Homography norm in pix, estimated on image with given width, height.
        """
        return self.dist(self.identity(), width, height)

    def dist_sourcespace(self, other, width=1, height=1):
        """
        Distance between homographies in pix, estimated on image with given
        width, height.
        (old formula ||B^{-1}A - I||, deprecated).
        """

        def distance_for_sample(h, x, y):
            vect = np.transpose(np.array([[x, y, 1]]))
            err = vect - np.dot(h, vect)
            error_norm = np.linalg.norm(err) / np.linalg.norm(vect)
            return error_norm

        diff = np.dot(np.linalg.inv(other.h), self.h)
        errors = []
        for corner_x in [0, width]:
            for corner_y in [0, height]:
                error = distance_for_sample(diff, corner_x, corner_y)
                errors.append(error)
        return max(errors)

    def dist(self, other, width=1, height=1):
        """
        Distance between homographies in pix, estimated on image with given
        width, height.
        """
        corners = np.array([
            [x, y] for x in [0, width] for y in [0, height]
        ])

        errors = np.linalg.norm(
            self(corners) - other(corners), axis=1)
        return np.max(errors)

    def rel_dist(self, other, width=1, height=1):
        """
        Distance between homographies, estimated on image with given size,
        and normalized by the diagonal length.
        """
        diag = np.linalg.norm([width, height])
        return self.dist(other, width, height) / diag

    def as_ndarray(self):
        return self.h

    def __str__(self):
        return self.h.__str__()

    def __repr__(self):
        return self.h.__repr__()

    def __getitem__(self, key):
        """ Returns submatrix, e.g. h[0, 1:2]. Usage as in ndarray. """
        return self.h[key]

    def __invert__(self):
        return Homography(np.linalg.inv(self.h))

    def __mul__(self, other):
        return Homography(np.dot(self.h, other.h))

    def __call__(self, pt):
        if isinstance(pt, shapely.geometry.Point):
            pt = np.array([pt.x, pt.y, 1.0])
        else:
            pt = np.asarray(pt, dtype=np.float64)
        if pt.shape[-1] == 2:
            pt = np.concatenate([pt, np.ones(pt.shape[:-1]+(1,))], -1)

        res = np.tensordot(pt, self.h, (-1, -1))
        return res[..., :2]/res[..., 2:3]


def from_points(src, dst):
    """
    Find homography that transforms four source points 'src' to destination
    points 'dst'.
    Both specified as 4x2 arrays, or lists of 4 shapely Point objects
    """
    assert len(src) == len(dst) == 4
    if hasattr(src[0], 'x'):
        src = np.array([[p.x, p.y] for p in src], dtype=np.float64)
    if hasattr(dst[0], 'x'):
        dst = np.array([[p.x, p.y] for p in dst], dtype=np.float64)
    # src_to_dst, _ = cv2.findHomography(src, dst)
    # src_to_dst = svd_homography(src, dst)
    src_to_dst = lstsq_homography(src, dst)
    return Homography(src_to_dst)


_src_mats = np.array([
    [[1, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 1]],
    [[0, 0, 0, 1, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 1]]
])
_src_mats8 = np.array([
    [[1, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 0, 0, 0, 0, 0, 1],
     [0, 0, 1, 0, 0, 0, 0, 0]],
    [[0, 0, 0, 1, 0, 0, 1, 0],
     [0, 0, 0, 0, 1, 0, 0, 1],
     [0, 0, 0, 0, 0, 1, 0, 0]]
])


def _stack1(a):
    return np.hstack([a, np.ones([len(a), 1])])


# does not depend on cv2
def svd_homography(src, dst):
    # Nx3  2x3x9 -> Nx2x9
    mat = np.dot(_stack1(src), _src_mats)
    mat[..., -3:] *= -dst[..., None]
    # solve homogeneous 8x9 system
    u, s, hv = np.linalg.svd(mat.reshape([-1, 9]))
    h = hv[-1].reshape([3, 3])
    return h/h[-1, -1]


# more robust, works for N>4
def lstsq_homography(src, dst):
    # [Nx3] x [2x3x8] -> [Nx2x8]
    mat = np.dot(_stack1(src), _src_mats8)
    mat[..., -2:] *= -dst[..., None]
    x, res, rank, s = np.linalg.lstsq(
        mat.reshape([-1, 8]), dst.flat)
    return np.concatenate([x, [1.0]]).reshape([3, 3])
