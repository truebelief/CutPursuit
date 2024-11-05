"""Test suite for Cut Pursuit algorithm."""

import unittest
import numpy as np
from cut_pursuit import CutPursuit, perform_cut_pursuit, decimate_pcd


class TestCutPursuit(unittest.TestCase):
    """Test cases for Cut Pursuit algorithm."""

    def setUp(self):
        """Set up test data."""
        self.n_points = 100
        self.points = np.random.rand(self.n_points, 3)
        self.cp = CutPursuit(self.n_points)

    def test_initialization(self):
        """Test CutPursuit initialization."""
        self.assertEqual(self.cp.n_vertex, self.n_points)
        self.assertEqual(self.cp.dim, 1)
        self.assertEqual(len(self.cp.vertex_weights), self.n_points + 2)

    def test_perform_cut_pursuit(self):
        """Test perform_cut_pursuit function."""
        K = 4
        lambda_ = 1.0
        components = perform_cut_pursuit(K, lambda_, self.points)
        self.assertEqual(len(components), len(self.points))

    def test_decimate_pcd(self):
        """Test point cloud decimation."""
        min_res = 0.1
        dec_idx, dec_inverse = decimate_pcd(self.points, min_res)
        self.assertLess(len(dec_idx), len(self.points))
        self.assertEqual(len(dec_inverse), len(self.points))


if __name__ == '__main__':
    unittest.main()