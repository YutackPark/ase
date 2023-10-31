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

"""
 Relax a structure
 Calculate its exact elasticity tensor
 Displace the structure slightly
 Initialize initial Hessian from elasticity tensor
 Optimize the cell
 Compare without setting the initial Hessian
"""

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
