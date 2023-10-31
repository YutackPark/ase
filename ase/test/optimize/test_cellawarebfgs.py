import pytest
from ase.optimize.cellawarebfgs import CellAwareBFGS
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter, UnitCellFilter
from ase.build import bulk, fcc110
from ase.calculators.emt import EMT
import numpy as np


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


def test_rattle_supercell():
    def relax(atoms):
        atoms.calc = EMT()
        relax = CellAwareBFGS(FrechetCellFilter(atoms, exp_cell_factor=1.0), alpha=70, long_output=True)
        relax.run(fmax=0.05, smax=0.005)
        return relax.nsteps

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
        print(atoms.get_forces())
        relax = CellAwareBFGS(FrechetCellFilter(atoms, exp_cell_factor=1.0), alpha=70, long_output=True)
        relax.run()
        steps.append(relax.nsteps)
    assert steps[0] == steps[1]
     

def oldcode():
    from ase.stress import get_elasticity_tensor
    import numpy as np
    from typing import Tuple

    from ase import Atoms
    from ase.build import bulk
    from ase.filters import ExpCellFilter
    from ase.optimize import BFGS
    from ase.io import read

    from run_optimization import run_optimization
    from hessian_initial import ExactHessianBFGS, CellBFGS

    from gpaw import GPAW, PW

    from ase.calculators.emt import EMT
    #def EMT():
    #    return GPAW(mode=PW(ecut=400), kpts=(4,4,4), txt='gpaw.txt')


    atoms = bulk('Au')
    atoms *= 5
    #atoms[0].symbol ='Al'
    from sys import argv

    alpha = 70

    #atoms = read(argv[1])
    #atoms.rattle(0.1)
    np.random.seed(1)
    atoms.calc = EMT()

    #relax = BFGS(atoms)
    #relax.run()

    #relax = BFGS(ExpCellFilter(atoms), alpha=alpha, logfile='/dev/null')
    relax = CellBFGS(atoms)
    run_optimization(relax)
    C_ijkl = get_elasticity_tensor(atoms)
    for i in range(3):
        for j in range(3):
            print(f'C_ijkl[{i}, {j}] =')
            for k in range(3):
                for l in range(3):
                    print(round(C_ijkl[i,j,k,l], 2), end=' ')
                print()
            print()
        print()

    d = 0.01
    deformation = np.eye(3) + d*(np.random.rand(3,3)-0.5)
    atoms.set_cell(atoms.get_cell() @ deformation, scale_atoms=True)

    #atoms *= (1,2,3)



    np.set_printoptions(precision=3, linewidth=200, suppress=True)

    atoms_cellbfgs = atoms.copy()
    atoms_cellbfgs.calc = EMT()
    relax_cellbfgs = CellBFGS(atoms_cellbfgs, alpha=alpha)
    print('Hessian initial guess tensor\n', np.round(relax_cellbfgs.H0[-9:, -9:],3))
    run_optimization(relax_cellbfgs)
    steps_cellbfgs = relax_cellbfgs.nsteps

    atoms_exact = atoms.copy()
    atoms_exact.calc = EMT()
    relax_exact = ExactHessianBFGS(atoms_exact, C_ijkl, alpha=alpha)
    print('Exact elasticity tensor\n', np.round(relax_exact.H0[-9:, -9:],3))
    run_optimization(relax_exact)
    steps_exact = relax_exact.nsteps

    if 1:
        atoms_copy = atoms.copy()
        atoms_copy.calc = EMT()
        relax = BFGS(ExpCellFilter(atoms_copy), logfile='/dev/null')
        relax.masks = [ np.ones( 3 * len(atoms) + 9) ]
        run_optimization(relax)
        steps_naive = relax.nsteps

    #run_optimization(relax)
    print(f'Naive {steps_naive} CellBFGS {steps_cellbfgs} Exact cell hessian {steps_exact}') 
