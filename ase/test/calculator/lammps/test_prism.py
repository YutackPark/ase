"""Test Prism"""
from math import sqrt

import numpy as np
from ase.calculators.lammps.coordinatetransform import calc_box_parameters


class TestCalcBoxParameters:
    """Test Prism"""
    def test_sc(self):
        """Test sc"""
        array = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        box = calc_box_parameters(array)
        box_ref = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(box, box_ref)

    def test_bcc(self):
        """Test bcc"""
        array = np.array([
            [-0.5, +0.5, +0.5],
            [+0.5, -0.5, +0.5],
            [+0.5, +0.5, -0.5],
        ])
        box = calc_box_parameters(array)
        box_ref = np.array([
            +0.8660254038,  # +sqrt(3) / 2
            +0.8164965809,  # +sqrt(6) / 3
            +0.7071067812,  # +sqrt(2) / 2
            -0.2886751346,  # -sqrt(3) / 6
            -0.2886751346,  # -sqrt(3) / 6
            -0.4082482905,  # -sqrt(6) / 6
        ])
        np.testing.assert_allclose(box, box_ref)

    def test_fcc(self):
        """Test fcc"""
        array = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ])
        box = calc_box_parameters(array)
        box_ref = np.array([
            +0.7071067812,  # +sqrt(2) / 2
            +0.6123724357,  # +sqrt(6) / 4
            +0.5773502692,  # +sqrt(3) / 3
            +0.3535533906,  # +sqrt(2) / 4
            +0.3535533906,  # +sqrt(2) / 4
            +0.2041241452,  # +sqrt(6) / 12
        ])
        np.testing.assert_allclose(box, box_ref)

    def test_hcp(self):
        """Test hcp"""
        covera = sqrt(8.0 / 3.0)
        array = np.array([
            [0.5, -0.5 * sqrt(3.0), 0.0],
            [0.5, +0.5 * sqrt(3.0), 0.0],
            [0.0, 0.0, covera],
        ])
        box = calc_box_parameters(array)
        box_ref = np.array([1.0, 0.5 * sqrt(3.0), covera, -0.5, 0.0, 0.0])
        np.testing.assert_allclose(box, box_ref)
