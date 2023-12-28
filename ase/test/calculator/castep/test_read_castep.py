"""Tests for the Castep.read method"""
from io import StringIO

import numpy as np
from ase.calculators.castep import _read_forces, _get_indices_to_sort_back
from ase.constraints import FixAtoms, FixCartesian


FORCES = """\
 ************************** Forces **************************
 *                                                          *
 *               Cartesian components (eV/A)                *
 * -------------------------------------------------------- *
 *                         x            y            z      *
 *                                                          *
 * Si              1     -0.02211     -0.02210     -0.02210 *
 * Si              2      0.02211      0.02210      0.02210 *
 *                                                          *
 ************************************************************
 """

CONSTRAINED_FORCES = """\
 ******************************** Constrained Forces ********************************
 *                                                                                  *
 *                           Cartesian components (eV/A)                            *
 * -------------------------------------------------------------------------------- *
 *                         x                    y                    z              *
 *                                                                                  *
 * Si              1      0.00000(cons'd)      0.00000(cons'd)      0.00000(cons'd) *
 * Si              2     -0.00252             -0.00252              0.00000(cons'd) *
 *                                                                                  *
 ************************************************************************************
"""  # noqa: E501


def test_forces():
    """Test if the Forces block can be parsed correctly."""
    out = StringIO(FORCES)
    out.readline()
    forces, constraints = _read_forces(out, n_atoms=2)
    forces_ref = [
        [-0.02211, -0.02210, -0.02210],
        [+0.02211, +0.02210, +0.02210],
    ]
    np.testing.assert_allclose(forces, forces_ref)
    assert not constraints


def test_constrainted_forces():
    """Test if the Constrainted Forces block can be parsed correctly."""
    out = StringIO(CONSTRAINED_FORCES)
    out.readline()
    forces, constraints = _read_forces(out, n_atoms=2)
    forces_ref = [
        [+0.00000, +0.00000, +0.00000],
        [-0.00252, -0.00252, +0.00000],
    ]
    constraints_ref = [
        FixAtoms(0),
        FixCartesian(1, mask=(0, 0, 1)),
    ]
    np.testing.assert_allclose(forces, forces_ref)
    assert all(constraints[0].index == constraints_ref[0].index)
    assert all(constraints[1].index == constraints_ref[1].index)
    assert all(constraints[1].mask == constraints_ref[1].mask)


def test_get_indices_to_sort_back():
    """Test if spicies in .castep are sorted back to atoms.symbols."""
    symbols = ['Si', 'Al', 'P', 'Al', 'P', 'Al', 'P', 'C']
    species = ['C', 'Al', 'Al', 'Al', 'Si', 'P', 'P', 'P']
    indices_ref = [4, 1, 5, 2, 6, 3, 7, 0]
    assert [species[_] for _ in indices_ref] == symbols
    indices = _get_indices_to_sort_back(symbols, species)
    assert indices.tolist() == indices_ref
