"""Tests for the Castep.read method"""
from io import StringIO

import numpy as np
from ase.calculators.castep import (
    _read_header,
    _read_forces,
    _read_stress,
    _read_mulliken_charges,
    _read_hirshfeld_details,
    _read_hirshfeld_charges,
    _get_indices_to_sort_back,
)
from ase.constraints import FixAtoms, FixCartesian
from ase.units import GPa

HEADER = """\
 ************************************ Title ************************************


 ***************************** General Parameters ******************************

 output verbosity                               : normal  (1)
 write checkpoint data to                       : castep.check
 type of calculation                            : single point energy
 stress calculation                             : off
 density difference calculation                 : off
 electron localisation func (ELF) calculation   : off
 Hirshfeld analysis                             : off
 polarisation (Berry phase) analysis            : off
 molecular orbital projected DOS                : off
 deltaSCF calculation                           : off
 unlimited duration calculation
 timing information                             : on
 memory usage estimate                          : on
 write extra output files                       : on
 write final potential to formatted file        : off
 write final density to formatted file          : off
 write BibTeX reference list                    : on
 write OTFG pseudopotential files               : on
 write electrostatic potential file             : on
 write bands file                               : on
 checkpoint writing                             : both castep_bin and check files
 random number generator seed                   :         42

 *********************** Exchange-Correlation Parameters ***********************

 using functional                               : Local Density Approximation
 DFT+D: Semi-empirical dispersion correction    : off

 ************************* Pseudopotential Parameters **************************

 pseudopotential representation                 : reciprocal space
 <beta|phi> representation                      : reciprocal space
 spin-orbit coupling                            : off

 **************************** Basis Set Parameters *****************************

 basis set accuracy                             : FINE
 finite basis set correction                    : none

 **************************** Electronic Parameters ****************************

 number of  electrons                           :  8.000
 net charge of system                           :  0.000
 treating system as non-spin-polarized
 number of bands                                :          8

 ********************* Electronic Minimization Parameters **********************

 Method: Treating system as metallic with density mixing treatment of electrons,
         and number of  SD  steps               :          1
         and number of  CG  steps               :          4

 total energy / atom convergence tol.           : 0.1000E-04   eV
 eigen-energy convergence tolerance             : 0.1000E-05   eV
 max force / atom convergence tol.              : ignored
 periodic dipole correction                     : NONE

 ************************** Density Mixing Parameters **************************

 density-mixing scheme                          : Broyden
 max. length of mixing history                  :         20

 *********************** Population Analysis Parameters ************************

 Population analysis with cutoff                :  3.000       A
 Population analysis output                     : summary and pdos components

 *******************************************************************************
"""  # noqa: E501

# Some keyword in the .param file triggers a more detailed header.
HEADER_DETAILED = """\
 ************************************ Title ************************************


 ***************************** General Parameters ******************************

 output verbosity                               : normal  (1)
 write checkpoint data to                       : castep.check
 type of calculation                            : single point energy
 stress calculation                             : off
 density difference calculation                 : off
 electron localisation func (ELF) calculation   : off
 Hirshfeld analysis                             : off
 polarisation (Berry phase) analysis            : off
 molecular orbital projected DOS                : off
 deltaSCF calculation                           : off
 unlimited duration calculation
 timing information                             : on
 memory usage estimate                          : on
 write extra output files                       : on
 write final potential to formatted file        : off
 write final density to formatted file          : off
 write BibTeX reference list                    : on
 write OTFG pseudopotential files               : on
 write electrostatic potential file             : on
 write bands file                               : on
 checkpoint writing                             : both castep_bin and check files

 output         length unit                     : A
 output           mass unit                     : amu
 output           time unit                     : ps
 output         charge unit                     : e
 output           spin unit                     : hbar/2
 output         energy unit                     : eV
 output          force unit                     : eV/A
 output       velocity unit                     : A/ps
 output       pressure unit                     : GPa
 output     inv_length unit                     : 1/A
 output      frequency unit                     : cm-1
 output force constant unit                     : eV/A**2
 output         volume unit                     : A**3
 output   IR intensity unit                     : (D/A)**2/amu
 output         dipole unit                     : D
 output         efield unit                     : eV/A/e
 output        entropy unit                     : J/mol/K
 output    efield chi2 unit                     : pm/V

 wavefunctions paging                           : none
 random number generator seed                   :   90945350
 data distribution                              : optimal for this architecture
 optimization strategy                          : balance speed and memory

 *********************** Exchange-Correlation Parameters ***********************

 using functional                               : Local Density Approximation
 relativistic treatment                         : Koelling-Harmon
 DFT+D: Semi-empirical dispersion correction    : off

 ************************* Pseudopotential Parameters **************************

 pseudopotential representation                 : reciprocal space
 <beta|phi> representation                      : reciprocal space
 spin-orbit coupling                            : off

 **************************** Basis Set Parameters *****************************

 plane wave basis set cut-off                   :   180.0000   eV
 size of standard grid                          :     1.7500
 size of   fine   gmax                          :    12.0285   1/A
 finite basis set correction                    : none

 **************************** Electronic Parameters ****************************

 number of  electrons                           :  8.000
 net charge of system                           :  0.000
 treating system as non-spin-polarized
 number of bands                                :          8

 ********************* Electronic Minimization Parameters **********************

 Method: Treating system as metallic with density mixing treatment of electrons,
         and number of  SD  steps               :          1
         and number of  CG  steps               :          4

 total energy / atom convergence tol.           : 0.1000E-04   eV
 eigen-energy convergence tolerance             : 0.1000E-05   eV
 max force / atom convergence tol.              : ignored
 convergence tolerance window                   :          3   cycles
 max. number of SCF cycles                      :         30
 number of fixed-spin iterations                :         10
 smearing scheme                                : Gaussian
 smearing width                                 : 0.2000       eV
 Fermi energy convergence tolerance             : 0.2721E-13   eV
 periodic dipole correction                     : NONE

 ************************** Density Mixing Parameters **************************

 density-mixing scheme                          : Broyden
 max. length of mixing history                  :         20
 charge density mixing amplitude                : 0.8000
 cut-off energy for mixing                      :  180.0       eV

 *********************** Population Analysis Parameters ************************

 Population analysis with cutoff                :  3.000       A
 Population analysis output                     : summary and pdos components

 *******************************************************************************
"""  # noqa: E501


def test_header():
    """Test if the header blocks can be parsed correctly."""
    out = StringIO(HEADER)
    parameters = _read_header(out)
    parameters_ref = {
        'task': 'SinglePoint',
        'iprint': 1,
        'calculate_stress': False,
        'xc_functional': 'LDA',
        'basis_precision': 'FINE',
        'finite_basis_corr': 0,
        'elec_energy_tol': 1e-5,
        'mixing_scheme': 'Broyden',
    }
    assert parameters == parameters_ref


def test_header_detailed():
    """Test if the header blocks can be parsed correctly."""
    out = StringIO(HEADER_DETAILED)
    parameters = _read_header(out)
    parameters_ref = {
        'task': 'SinglePoint',
        'iprint': 1,
        'calculate_stress': False,
        'opt_strategy': 'Default',
        'xc_functional': 'LDA',
        'cut_off_energy': 180.0,
        'finite_basis_corr': 0,
        'elec_energy_tol': 1e-5,
        'elec_convergence_win': 3,
        'mixing_scheme': 'Broyden',
    }
    assert parameters == parameters_ref


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


STRESS = """\
 ***************** Stress Tensor *****************
 *                                               *
 *          Cartesian components (GPa)           *
 * --------------------------------------------- *
 *             x             y             z     *
 *                                               *
 *  x     -0.006786     -0.035244      0.023931  *
 *  y     -0.035244     -0.006786      0.023931  *
 *  z      0.023931      0.023931     -0.011935  *
 *                                               *
 *  Pressure:    0.0085                          *
 *                                               *
 *************************************************
"""


def test_stress():
    """Test if the Stress Tensor block can be parsed correctly."""
    out = StringIO(STRESS)
    out.readline()
    results = _read_stress(out)
    stress_ref = [
        [-0.006786, -0.035244, +0.023931],
        [-0.035244, -0.006786, +0.023931],
        [+0.023931, +0.023931, -0.011935],
    ]
    stress_ref = np.array(stress_ref) * GPa
    pressure_ref = 0.0085 * GPa
    np.testing.assert_allclose(results['stress'], stress_ref)
    np.testing.assert_allclose(results['pressure'], pressure_ref)


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
    out.readline()  # read header
    results = _read_mulliken_charges(out)
    np.testing.assert_allclose(results['charges'], [+0.704, -0.704])
    assert 'magmoms' not in results


def test_mulliken_spin_polarized():
    """Test if the Atomic Populations block can be parsed correctly."""
    out = StringIO(MULLIKEN_SPIN_POLARIZED)
    out.readline()  # read header
    results = _read_mulliken_charges(out)
    np.testing.assert_allclose(results['charges'], [-0.114, +0.114])
    np.testing.assert_allclose(results['magmoms'], [+4.785, -0.027])


HIRSHFELD_DETAILS = """\
 Species     1,  Atom     1  :  Al
  Fractional coordinates :
                                        0.000000000   0.000000000   0.000000000
  Cartesian coordinates (A) :
                                        0.000000000   0.000000000   0.000000000
  Free atom total nuclear charge (e) :
                                        3.000000000
  Free atom total electronic charge on real space grid (e) :
                                       -3.000000000
  SCF total electronic charge on real space grid (e) :
                                       -8.000000000
  cut-off radius for r-integrals :
                                       10.000000000
  Free atom volume (Bohr**3) :
                                       67.048035000
  Hirshfeld total electronic charge (e) :
                                       -2.821040742
  Hirshfeld net atomic charge (e) :
                                        0.178959258
  Hirshfeld atomic volume (Bohr**3) :
                                       61.353953500
  Hirshfeld / free atomic volume :
                                        0.915074595

 Species     2,  Atom     1  :  P
  Fractional coordinates :
                                        0.250000000   0.250000000   0.250000000
  Cartesian coordinates (A) :
                                        1.357500000   1.357500000   1.357500000
  Free atom total nuclear charge (e) :
                                        5.000000000
  Free atom total electronic charge on real space grid (e) :
                                       -5.000000000
  SCF total electronic charge on real space grid (e) :
                                       -8.000000000
  cut-off radius for r-integrals :
                                       10.000000000
  Free atom volume (Bohr**3) :
                                       70.150468179
  Hirshfeld total electronic charge (e) :
                                       -5.178959258
  Hirshfeld net atomic charge (e) :
                                       -0.178959258
  Hirshfeld atomic volume (Bohr**3) :
                                       66.452900385
  Hirshfeld / free atomic volume :
                                        0.947290904

"""


def test_hirshfeld_details():
    """Test if the Hirshfeld block of ispin > 1 can be parsed correctly."""
    out = StringIO(HIRSHFELD_DETAILS)
    results = _read_hirshfeld_details(out, 2)
    np.testing.assert_allclose(
        results['hirshfeld_volume_ratios'],
        [0.915074595, 0.947290904],
    )


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
