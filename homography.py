"""
library for 2d homographies.

The Homography object represents a 2D homography as a 3x3 matrix.
Homographies can be applied directly on numpy arrays or Shapely points using
the "call operator" (brackets), composed using ``*`` and inverted using ``~``.

This module supports basic operations, conversion methods and utilities.
Sample usage:

>>> h = Homography.translation(5, -1) * Homography.rotation(90)
>>> h.as_ndarray().astype(int)
array([[ 0,  1,  5],
       [-1,  0, -1],
       [ 0,  0,  1]])

>>> print(h([[0, 0],[1, 0],[1 ,1]]))
[[ 5. -1.]
 [ 5. -2.]
 [ 6. -2.]]

>>> print(~h*h)
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
>>> print((~h)(h([12, 34])))
[12. 34.]

"""
import affine
import numpy as np

try:
    import cv2
    no_cv2 = False
except ImportError:
    no_cv2 = True


class Homography(object):
    """
    The 3x3 homography transformation matrix contains 8 free parameters and
    represents transfromation from :math:`(x,y)` to :math:`(x',y')`
    as follows::

        | x0 |   | a  b  c | | x |
        | y0 | = | d  e  f | | y |
        | z0 |   | g  h  1 | | 1 |

      x' = x0 / z0
      y' = y0 / z0
    """

    def __init__(self, other=None):
        """
        Constructs itself from Homography, Affine or 3x3 mat.
        The default constuctor returns the identity homography.
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
        :param affine.Affine aff: the affine to convert from
        :rtype: Homography
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
        Distance between homographies in source space, estimated on output
        image with given width, height.
        """
        invself, invother = ~self, ~other
        return invself.dist(invother, width, height)

    def dist(self, other, width=1, height=1):
        """
        Distance between homographies in output space, estimated on image
        with given width, height.
        """
        corners = np.array([
            [x, y] for x in [0, width] for y in [0, height]
        ])
        errors = np.linalg.norm(
            self(corners) - other(corners), axis=1)
        return np.max(errors)

    def dist_bidirectional(self, other, width=1, height=1):
        """
        Distance between homographies as max between distance in source and
        image space.
        Estimated on image with given width, height.
        """
        d1 = self.dist(other, width, height)
        d2 = self.dist_sourcespace(other, width, height)
        return max(d1, d2)

    def rel_dist(self, other, width=1, height=1):
        """
        Distance between homographies, estimated on image with given size,
        and normalized by the diagonal length.
        """
        diag = np.linalg.norm([width, height])
        return self.dist(other, width, height) / diag

    def get_shift_at_point(self, point):
        """
        Calculates the shift applied to the specified point when the homography is applied. Input point can be a 1D 
        2 element numpy array or a Shapely point
        """
        point = adapt_point_input(point)
        return self(point) - point

    def as_ndarray(self):
        return self.h

    def __str__(self):
        return self.h.__str__()

    def __repr__(self):
        return self.h.__repr__()

    def __getitem__(self, key):
        """
        Returns submatrix, e.g.:

        >>> print(Homography.identity()[2, 1:])
        [0. 1.]
        """
        return self.h[key]

    def __invert__(self):
        return Homography(np.linalg.inv(self.h))

    def __mul__(self, other):
        return Homography(np.dot(self.h, other.h))

    def __call__(self, point):
        point = adapt_point_input(point)
        if point.shape[-1] == 2:
            point = np.concatenate([point, np.ones(point.shape[:-1]+(1,))], -1)
        res = np.tensordot(point, self.h, (-1, -1))
        return res[..., :2]/res[..., 2:3]


def adapt_point_input(point):
    """
    Checks if the point is a Shapely point (by looking at its attributes) and converts it to a 1D 2 element numpy array
    """
    if hasattr(point, 'x') and hasattr(point, 'y'):
        point = np.array([point.x, point.y, 1.0])
    else:
        point = np.asarray(point, dtype=np.float64)
    
    return point


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
    src_to_dst, mask = cv2.findHomography(src, dst)
    return Homography(src_to_dst)
