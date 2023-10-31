import pytest
import numpy as np
from ase.optimize.cellawarebfgs import CellAwareBFGS
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter, UnitCellFilter
from ase.build import bulk, fcc110
from ase.calculators.emt import EMT
from ase.stress import get_elasticity_tensor




def test_rattle_supercell_old():
    def relax(atoms):
        atoms.calc = EMT()
        relax = BFGS(FrechetCellFilter(atoms), alpha=70)
        relax.run(fmax=0.05)
        return relax.nsteps

    atoms = bulk('Au')
    atoms *= 2
    atoms.rattle(0.05)
    nsteps = relax(atoms.copy())
    atoms *= 2
    nsteps2 = relax(atoms.copy())
    assert nsteps != nsteps2


def relax(atoms):
    atoms.calc = EMT()
    relax = CellAwareBFGS(FrechetCellFilter(atoms, exp_cell_factor=1.0), alpha=70, long_output=True)
    relax.run(fmax=0.00005, smax=0.0000005)
    return relax.nsteps


def test_rattle_supercell():
    atoms = bulk('Au')
    atoms *= 2
    atoms.rattle(0.05)
    nsteps = relax(atoms.copy())
    atoms *= 2
    nsteps2 = relax(atoms.copy())
    assert nsteps == nsteps2


@pytest.mark.parametrize('filt', [FrechetCellFilter, UnitCellFilter])
def test_cellaware_bfgs_2d(filt):
    atoms = fcc110('Au', size=(2,2, 3), vacuum=4)
    orig_cell = atoms.cell.copy()
    atoms.cell = atoms.cell @ np.array([[1.0, 0.05, 0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    atoms.calc = EMT()
    relax = CellAwareBFGS(filt(atoms, mask=[1,1,0,0,0,1]), alpha=70, long_output=True)
    relax.run(fmax=0.0005)
    assert np.allclose(atoms.cell[2,:], orig_cell[2,:])
    assert np.allclose(atoms.cell[:, 2], orig_cell[:, 2])


def test_cellaware_bfgs():
    steps = []
    for scale in [1, 2]:
        atoms = bulk('Au')
        atoms *= scale
        atoms.calc = EMT()
        relax = CellAwareBFGS(FrechetCellFilter(atoms, exp_cell_factor=1.0), alpha=70, long_output=True)
        relax.run()
        steps.append(relax.nsteps)
    assert steps[0] == steps[1]

def test_elasticity_tensor():
    atoms = bulk('Au')
    atoms *= 2
    atoms.calc = EMT()
    relax(atoms)
    C_ijkl = get_elasticity_tensor(atoms, verbose=True)


    d = 0.01
    deformation = np.eye(3) + d*(np.random.rand(3,3)-0.5)
    atoms.set_cell(atoms.get_cell() @ deformation, scale_atoms=True)
   
    def ExactHessianBFGS(atoms, C_ijkl, alpha=70):
        atoms_and_cell = FrechetCellFilter(atoms, exp_cell_factor=1.0)
        relax = CellAwareBFGS(atoms_and_cell, alpha=70, long_output=True)
        C_ijkl = C_ijkl.copy()
        # Supplement the tensor with suppression of pure rotations (which are right now 0 eigenvalues)
        # Loop over all basis vectors of skew symmetric real matrix
        for i,j in ((0,1),(0,2),(1,2)):
            Q = np.zeros((3,3))
            Q[i,j], Q[j,i] = 1, -1
            C_ijkl += np.einsum('ij,kl->ijkl', Q, Q) * alpha / 2
        relax.H0[-9:, -9:] = C_ijkl.reshape((9,9)) * atoms.cell.volume
        return relax

    rlx = ExactHessianBFGS(atoms, C_ijkl)
    rlx.run(fmax=0.05, smax=0.005)
    assert rlx.nsteps == 1

    # Make sure we can approximate the elasticity tensor within 10%
    # using the CellAwareBFGS
    tmp = CellAwareBFGS(FrechetCellFilter(atoms, exp_cell_factor=1.0), bulk_modulus=175, poisson_ratio=0.46)
    for a, b in zip(rlx.H0[-9:, -9:].ravel(), tmp.H0[-9:, -9:].ravel()):
        if abs(a) > 0.001:
            print(a, b)
            assert np.abs((a-b) / a) < 0.1
