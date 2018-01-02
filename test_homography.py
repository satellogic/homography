import unittest

import numpy as np
from affine import Affine
from shapely.geometry import Point

import homography
from homography import Homography

try:
    no_cv2 = False
    import cv2
except ImportError:
    no_cv2 = True


class TestHomography(unittest.TestCase):
    def test_constructor(self):
        """ verify all construction methods are identical. """
        h_from_ndarray = Homography([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        h_from_affine = Homography(Affine.identity())
        h_from_homography = Homography(h_from_ndarray)

        self.assertEqual(h_from_ndarray, h_from_affine)
        self.assertEqual(h_from_ndarray, h_from_homography)
        self.assertEqual(h_from_ndarray, Homography.identity())

    def test_arithmetics(self):
        h1 = Homography.rotation(90)
        h2 = Homography.translation(3, 4)
        h3 = Homography.scale(5, 6)
        h = h3 * h2 * h1

        actual = h([7, 8])
        self.assertTrue(np.allclose(
            actual, [5 * (8 + 3), 6 * (-7 + 4)])
        )

        via_3d = h.apply(7, 8) / h.apply(7, 8)[2]
        self.assertTrue(np.allclose(actual, via_3d[:2]))

    def test_projectivity(self):
        h = Homography.translation(3, 4)
        self.assertAlmostEqual(h.projectivity(), 0)

    def test_distance(self):
        h1 = Homography.identity()
        h2 = Homography.translation(3, 4)

        img_size = [42, 123456]
        dist = h1.dist_sourcespace(h2, *img_size)
        expected_dist = np.sqrt(3**2 + 4**2)
        self.assertAlmostEqual(expected_dist, dist)
        dist = h1.dist(h2, *img_size)
        self.assertAlmostEqual(expected_dist, dist)

        expected_rel_dist = expected_dist / np.linalg.norm(img_size)
        rel_dist = h1.rel_dist(h2, *img_size)
        self.assertAlmostEqual(expected_rel_dist, rel_dist)

        self.assertAlmostEqual(h1.norm(42, 123456), 0)
        self.assertAlmostEqual(h2.norm(42, 123456), expected_dist)

    def test_affine(self):
        affine = Affine.translation(42, 142)
        actual = Homography.from_affine(affine)
        expected = Homography.translation(42, 142)
        self.assertAlmostEqual(actual, expected)

        as_affine = actual.to_affine()
        self.assertAlmostEqual(affine, as_affine)

    def test_todict(self):
        h1 = Homography.translation(2, 3)*Homography.scale(2, 1)
        h2 = Homography.from_dict(h1.to_dict())
        self.assertEqual(h1, h2)

    def test_equal(self):
        identity = Homography.identity()
        eps = 1e-7
        shift_by_eps = Homography.translation(eps, eps)
        self.assertTrue(shift_by_eps.equal(identity, 2 * eps))
        self.assertFalse(shift_by_eps.equal(identity, eps / 2.))
        self.assertAlmostEqual(shift_by_eps, identity)

    @unittest.skipIf(no_cv2, 'Skipped since couldnt import opencv2.')
    def test_from_points(self):
        src = np.array(
            [[0.0, 0.0], [200.0, 0.0], [200.0, 100.0], [0.0, 100.0]])
        diff = np.array([10, 20])
        dst = src - diff
        h = homography.from_points(src, dst)
        expected = Homography.translation(-diff[0], -diff[1])
        self.assertEqual(h, expected)
        src = [Point(x, y) for x, y in src]
        dst = [Point(x, y) for x, y in dst]
        h = homography.from_points(src, dst)
        self.assertEqual(h, expected)
