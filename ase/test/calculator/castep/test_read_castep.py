"""Tests for the Castep.read method"""
from io import StringIO

import numpy as np
from ase.calculators.castep import (
    _read_forces,
    _read_mulliken_charges,
    _read_hirshfeld_charges,
    _get_indices_to_sort_back,
)
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


# bulk("AlP", "zincblende", a=5.43)
MULLIKEN_SPIN_UNPOLARIZED = """\
     Atomic Populations (Mulliken)
     -----------------------------
Species          Ion     s       p       d       f      Total   Charge (e)
==========================================================================
  Al              1     0.935   1.361   0.000   0.000   2.296     0.704
  P               1     1.665   4.039   0.000   0.000   5.704    -0.704
==========================================================================
"""

# bulk("MnTe", "zincblende", a=6.34)
MULLIKEN_SPIN_POLARIZED = """\
     Atomic Populations (Mulliken)
     -----------------------------
Species          Ion Spin      s       p       d       f      Total   Charge(e)   Spin(hbar/2)
==============================================================================================
  Mn              1   up:     1.436   3.596   4.918   0.000   9.950    -0.114        4.785
                  1   dn:     1.293   3.333   0.538   0.000   5.164
  Te              1   up:     0.701   2.229   0.000   0.000   2.929     0.114       -0.027
                  1   dn:     0.763   2.194   0.000   0.000   2.956
==============================================================================================
"""  # noqa: E501


def test_mulliken_spin_unpolarized():
    """Test if the Atomic Populations block can be parsed correctly."""
    out = StringIO(MULLIKEN_SPIN_UNPOLARIZED)
    for _ in range(3):
        out.readline()
    charges, magmoms = _read_mulliken_charges(out, spin_polarized=False)
    np.testing.assert_allclose(charges, [+0.704, -0.704])
    np.testing.assert_allclose(magmoms, [])


def test_mulliken_spin_polarized():
    """Test if the Atomic Populations block can be parsed correctly."""
    out = StringIO(MULLIKEN_SPIN_POLARIZED)
    for _ in range(3):
        out.readline()
    charges, magmoms = _read_mulliken_charges(out, spin_polarized=True)
    np.testing.assert_allclose(charges, [-0.114, +0.114])
    np.testing.assert_allclose(magmoms, [+4.785, -0.027])


HIRSHFELD_SPIN_UNPOLARIZED = """\
     Hirshfeld Analysis
     ------------------
Species   Ion     Hirshfeld Charge (e)
======================================
  Al       1                 0.18
  P        1                -0.18
======================================
"""

HIRSHFELD_SPIN_POLARIZED = """\
     Hirshfeld Analysis
     ------------------
Species   Ion     Hirshfeld Charge (e)  Spin (hbar/2)
===================================================
  Mn       1                 0.06        4.40
  Te       1                -0.06        0.36
===================================================
"""


def test_hirshfeld_spin_unpolarized():
    """Test if the Hirshfeld Analysis block can be parsed correctly."""
    out = StringIO(HIRSHFELD_SPIN_UNPOLARIZED)
    out.readline()  # read header
    results = _read_hirshfeld_charges(out)
    np.testing.assert_allclose(results['hirshfeld_charges'], [+0.18, -0.18])
    assert 'hirshfeld_magmoms' not in results


def test_hirshfeld_spin_polarized():
    """Test if the Hirshfeld Analysis block can be parsed correctly."""
    out = StringIO(HIRSHFELD_SPIN_POLARIZED)
    out.readline()  # read header
    results = _read_hirshfeld_charges(out)
    np.testing.assert_allclose(results['hirshfeld_charges'], [+0.06, -0.06])
    np.testing.assert_allclose(results['hirshfeld_magmoms'], [+4.40, +0.36])


def test_get_indices_to_sort_back():
    """Test if spicies in .castep are sorted back to atoms.symbols."""
    symbols = ['Si', 'Al', 'P', 'Al', 'P', 'Al', 'P', 'C']
    species = ['C', 'Al', 'Al', 'Al', 'Si', 'P', 'P', 'P']
    indices_ref = [4, 1, 5, 2, 6, 3, 7, 0]
    assert [species[_] for _ in indices_ref] == symbols
    indices = _get_indices_to_sort_back(symbols, species)
    assert indices.tolist() == indices_ref
