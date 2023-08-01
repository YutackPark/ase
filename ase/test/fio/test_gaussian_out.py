"""Tests for the gaussian-out format."""
from io import StringIO

import numpy as np
import pytest

from ase.io import read
from ase.io.formats import match_magic
from ase import units


BUF_H2O = r"""
 Entering Gaussian System, Link 0=g16

...

 ******************************************
 Gaussian 16:  ES64L-G16RevA.03 25-Dec-2016
                 6-Apr-2021
 ******************************************

...

                          Input orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          8           0        1.1            2.2        3.3
      2          1           0        4.4            5.5        6.6
      3          1           0        7.7            8.8        9.9
 ---------------------------------------------------------------------

...

 SCF Done:  E(RB3LYP) =  -12.3456789     A.U. after    9 cycles

...

 -------------------------------------------------------------------
 Center     Atomic                   Forces (Hartrees/Bohr)
 Number     Number              X              Y              Z
 -------------------------------------------------------------------
      1        8              0.1              0.2            0.3
      2        1              0.4              0.5            0.6
      3        1              0.7              0.8            0.9
 -------------------------------------------------------------------
"""

BUF_F2_RHF = r"""
 Entering Gaussian System, Link 0=g16
...
                          Input orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          9           0        0.000000    0.000000    0.700000
      2          9           0        0.000000    0.000000   -0.700000
 ---------------------------------------------------------------------
...
 SCF Done:  E(RHF) =  -198.700044583     A.U. after    8 cycles
"""

BUF_F2_MP2 = BUF_F2_RHF + r"""
...
 E2 =    -0.4264521750D+00 EUMP2 =    -0.19912649675787D+03
"""

BUF_F2_CCSD = BUF_F2_MP2 + r"""
...
 Wavefunction amplitudes converged. E(Corr)=     -199.13391098
"""

BUF_F2_CCSD_T = BUF_F2_CCSD + r"""
...
 CCSD(T)= -0.19914648303D+03
"""


def test_match_magic():
    """Test if the file type can be guessed correctly."""
    bytebuf = BUF_H2O.encode('ascii')
    assert match_magic(bytebuf).name == 'gaussian-out'


def test_gaussian_out():
    """Test if positions, energy, and forces are parsed correctly."""
    atoms = read(StringIO(BUF_H2O), format='gaussian-out')
    assert str(atoms.symbols) == 'OH2'
    assert atoms.positions == pytest.approx(np.array([
        [1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6],
        [7.7, 8.8, 9.9],
    ]))
    assert not any(atoms.pbc)
    assert atoms.cell.rank == 0

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert energy / units.Ha == pytest.approx(-12.3456789)
    assert forces / (units.Ha / units.Bohr) == pytest.approx(np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]))


def test_mp2():
    """Test if the MP2 energy is parsed correctly."""
    atoms = read(StringIO(BUF_F2_MP2), format="gaussian-out")
    energy = atoms.get_potential_energy()
    assert energy / units.Ha == pytest.approx(-0.19912649675787e+03)


def test_ccsd():
    """Test if the CCSD energy is parsed correctly."""
    atoms = read(StringIO(BUF_F2_CCSD), format="gaussian-out")
    energy = atoms.get_potential_energy()
    assert energy / units.Ha == pytest.approx(-199.13391098)


def test_ccsd_t():
    """Test if the CCSD(T) energy is parsed correctly."""
    atoms = read(StringIO(BUF_F2_CCSD_T), format="gaussian-out")
    energy = atoms.get_potential_energy()
    assert energy / units.Ha == pytest.approx(-0.19914648303e+03)
