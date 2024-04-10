"""Tests for the gaussian-out format."""
from io import StringIO

import numpy as np
import pytest

from ase import units
from ase.io import read
from ase.io.formats import match_magic

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
      1          8           0        0.000000    0.000000    0.119262
      2          1           0        0.000000    0.763239   -0.477047
      3          1           0        0.000000   -0.763239   -0.477047
 ---------------------------------------------------------------------

...

 SCF Done:  E(RHF) =  -75.9834173665     A.U. after   10 cycles
"""

BUF_H2O_L601 = r"""
 (Enter /opt/bwhpc/common/chem/gaussian/g16.C.01/x86_64-Intel-avx2-source/g16/l601.exe)
...
 Dipole moment (field-independent basis, Debye):
    X=              0.0000    Y=             -0.0000    Z=             -2.6431  Tot=              2.6431
"""  # noqa: E501

BUF_H2O_L716 = r"""
 (Enter /opt/bwhpc/common/chem/gaussian/g16.C.01/x86_64-Intel-avx2-source/g16/l716.exe)
 Dipole        = 3.27065103D-16-1.33226763D-15-1.03989005D+00
 -------------------------------------------------------------------
 Center     Atomic                   Forces (Hartrees/Bohr)
 Number     Number              X              Y              Z
 -------------------------------------------------------------------
      1        8          -0.000000000   -0.000000000   -0.036558637
      2        1          -0.000000000   -0.003968101    0.018279318
      3        1           0.000000000    0.003968101    0.018279318
 -------------------------------------------------------------------
"""  # noqa: E501

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

BUF_H2O_OPT = r""" Entering Gaussian System, Link 0=g16
 Initial command:
...
(Enter /soft/gaussian/16-c.02/g16/l202.exe)
                          Input orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          8           0        0.000000    0.000000    0.117782
      2          1           0        0.000000    0.756577   -0.476307
      3          1           0       -0.000000   -0.756577   -0.476307
 ---------------------------------------------------------------------
                    Distance matrix (angstroms):
                    1          2          3
     1  O    0.000000
     2  H    0.961952   0.000000
     3  H    0.961952   1.513154   0.000000
 Stoichiometry    H2O
 Framework group  C2V[C2(O),SGV(H2)]
 Deg. of freedom     2
 Full point group                 C2V     NOp   4
 RotChk:  IX=0 Diff= 3.20D-16
 Largest Abelian subgroup         C2V     NOp   4
 Largest concise Abelian subgroup C2      NOp   2
                         Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          8           0        0.000000    0.000000    0.118818
      2          1           0       -0.000000    0.756577   -0.475272
      3          1           0       -0.000000   -0.756577   -0.475272
 ---------------------------------------------------------------------

 ...
SCF Done:  E(RB3LYP) =  -76.4259945508     A.U. after    9 cycles

***** Axes restored to original set *****
 -------------------------------------------------------------------
 Center     Atomic                   Forces (Hartrees/Bohr)
 Number     Number              X              Y              Z
 -------------------------------------------------------------------
      1        8          -0.000000000    0.000000000   -0.006569831
      2        1          -0.000000000   -0.005632960    0.003284915
      3        1           0.000000000    0.005632960    0.003284915
 -------------------------------------------------------------------

 ...

 (Enter /soft/gaussian/16-c.02/g16/l202.exe)
                          Input orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          8           0       -0.000000   -0.000000    0.117858
      2          1           0        0.000000    0.756683   -0.476345
      3          1           0       -0.000000   -0.756683   -0.476345
 ---------------------------------------------------------------------
                    Distance matrix (angstroms):
                    1          2          3
     1  O    0.000000
     2  H    0.962105   0.000000
     3  H    0.962105   1.513366   0.000000
 Stoichiometry    H2O
 Framework group  C2V[C2(O),SGV(H2)]
 Deg. of freedom     2
 Full point group                 C2V     NOp   4
 RotChk:  IX=0 Diff= 4.65D-16
 Largest Abelian subgroup         C2V     NOp   4
 Largest concise Abelian subgroup C2      NOp   2
                         Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          8           0       -0.000000    0.000000    0.118841
      2          1           0        0.000000    0.756683   -0.475362
      3          1           0       -0.000000   -0.756683   -0.475362
 ---------------------------------------------------------------------

 ...

 SCF Done:  E(RB3LYP) =  -76.4260779687     A.U. after    7 cycles

 ...

***** Axes restored to original set *****
 -------------------------------------------------------------------
 Center     Atomic                   Forces (Hartrees/Bohr)
 Number     Number              X              Y              Z
 -------------------------------------------------------------------
      1        8          -0.000000000   -0.000000000    0.000176354
      2        1           0.000000000    0.000122881   -0.000088177
      3        1           0.000000000   -0.000122881   -0.000088177
 -------------------------------------------------------------------
 Cartesian Forces:  Max     0.000176354 RMS     0.000092406

"""


def test_match_magic():
    """Test if the file type can be guessed correctly."""
    bytebuf = BUF_H2O.encode('ascii')
    assert match_magic(bytebuf).name == 'gaussian-out'


def test_gaussian_out_l601():
    """Test if positions and energy are parsed correctly.

    Test also if dipole moment is parsed correctly from `l601.exe`.
    This corresponds to the options without `Forces` and `Pop=None`.
    """
    atoms = read(StringIO(BUF_H2O + BUF_H2O_L601), format='gaussian-out')
    assert str(atoms.symbols) == 'OH2'
    assert atoms.positions == pytest.approx(np.array([
        [+0.000000, +0.000000, +0.119262],
        [+0.000000, +0.763239, -0.477047],
        [+0.000000, -0.763239, -0.477047],
    ]))
    assert not any(atoms.pbc)
    assert atoms.cell.rank == 0

    energy = atoms.get_potential_energy()
    assert energy / units.Ha == pytest.approx(-75.9834173665)

    dipole_moment_ref = pytest.approx(np.array([+0.0000, -0.0000, -2.6431]))
    assert atoms.get_dipole_moment() / units.Debye == dipole_moment_ref


def test_gaussian_out_l716():
    """Test if forces and dipole moment are parsed correctly from `l716.exe`.

    This corresponds to the options with `Forces` and `Pop=None`.
    """
    atoms = read(StringIO(BUF_H2O + BUF_H2O_L716), format='gaussian-out')
    forces = atoms.get_forces()
    assert forces / (units.Ha / units.Bohr) == pytest.approx(np.array([
        [-0.000000000, -0.000000000, -0.036558637],
        [-0.000000000, -0.003968101, +0.018279318],
        [+0.000000000, +0.003968101, +0.018279318],
    ]))
    assert atoms.get_dipole_moment() / units.Bohr == pytest.approx(np.array(
        [+3.27065103e-16, -1.33226763e-15, -1.03989005e+00],
    ))


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


def test_gaussian_opt():
    """Test if we determine the correct number of geometries in the output
    produced during optimization"""
    atoms = read(StringIO(BUF_H2O_OPT), format="gaussian-out", index=':')
    assert len(atoms) == 2

    # Ensure the energy and forces were parsed and differ
    assert atoms[0].get_potential_energy() != atoms[1].get_potential_energy()
    assert not np.isclose(atoms[0].get_forces(), atoms[1].get_forces()).all()
