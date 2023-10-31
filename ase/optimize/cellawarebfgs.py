from typing import IO, Optional, Union
import numpy as np
from ase.filters import ExpCellFilter
from ase.optimize import BFGS
from ase import Atoms


def calculate_isotropic_elasticity_tensor(bulk_modulus, poisson_ratio, suppress_rotation=0):
    """
    Parameters:
           bulk_modulus  Bulk Modulus of the isotropic system used to set up
                         the Hessian (in GPa). Default value 500 GPa corresponds
                         approximately to diamond.

           poisson_ratio Poisson ratio of the isotropic system ysed to set up
                         the initial Hessian (unitless, between -1 and 0.5).
                         Default value is 0.3.

           suppress_rotation The rank-2 matrix C_ijkl.reshape((9,9)) has by default
                             6 non zero eigenvalues, because energy is invariant 
                             to ortonormal rotations of the cell vector.
                             This serves as a bad initial Hessian due to 3 zero eigenvalues.
                             Suppress rotation sets a value for those zero eigenvalues.

           Returns C_ijkl
    """

    bulk_modulus = 0.00624150907 * bulk_modulus # GPa to eV / Å^3
    g = np.eye(3)

    # https://scienceworld.wolfram.com/physics/LameConstants.html
    _lambda = 3 * bulk_modulus * (poisson_ratio) / (1 + 1 * poisson_ratio)
    _mu = _lambda * (1 - 2 * poisson_ratio) / (2 * poisson_ratio)

    # https://en.wikipedia.org/wiki/Elasticity_tensor
    g_ij = np.eye(3)

    # Construct 4th rank Elasticity tensor for isotropic systems
    C_ijkl = _lambda * np.einsum('ij,kl->ijkl',g_ij,g_ij)
    C_ijkl += _mu * (np.einsum('ik,jl->ijkl', g_ij, g_ij) +
                     np.einsum('il,kj->ijkl', g_ij, g_ij))

    # Supplement the tensor with suppression of pure rotations (which are right now 0 eigenvalues)
    # Loop over all basis vectors of skew symmetric real matrix
    for i,j in ((0,1),(0,2),(1,2)):
        Q = np.zeros((3,3))
        Q[i,j], Q[j,i] = 1, -1
        C_ijkl += np.einsum('ij,kl->ijkl', Q, Q) * suppress_rotation / 2

    return C_ijkl


class CellAwareBFGS(BFGS):
    def __init__(
        self,
        atoms: Atoms,
        restart: Optional[str] = None,
        logfile: Union[IO, str] = '-',
        trajectory: Optional[str] = None,
        maxstep: Optional[float] = None,
        maxcellstep: Optional[float] = None,
        master: Optional[bool] = None,
        bulk_modulus: Optional[float] = 145,
        poisson_ratio: Optional[float] = 0.3,
        alpha: Optional[float] = None,
    ):
        self.bulk_modulus = bulk_modulus
        self.poisson_ratio = poisson_ratio
        BFGS.__init__(self, atoms, restart, logfile,
                      trajectory,
                      maxstep, master, alpha)
        assert not isinstance(atoms, Atoms)
    
    def initialize(self):
        BFGS.initialize(self)
        C_ijkl = calculate_isotropic_elasticity_tensor(
            self.bulk_modulus,
            self.poisson_ratio,
            suppress_rotation=self.alpha
            )
        self.H0[-9:, -9:] = C_ijkl.reshape((9,9)) * self.atoms.atoms.cell.volume


def ExactHessianBFGS(atoms, C_ijkl, alpha=70):
    atoms_and_cell = ExpCellFilter(atoms)
    relax = BFGS(atoms_and_cell, alpha=alpha)
    C_ijkl = C_ijkl.copy()
    # Supplement the tensor with suppression of pure rotations (which are right now 0 eigenvalues)
    # Loop over all basis vectors of skew symmetric real matrix
    for i,j in ((0,1),(0,2),(1,2)):
        Q = np.zeros((3,3))
        Q[i,j], Q[j,i] = 1, -1
        C_ijkl += np.einsum('ij,kl->ijkl', Q, Q) * alpha / 2
    relax.H0[-9:, -9:] = C_ijkl.reshape((9,9)) * atoms.cell.volume
    relax.masks = [ np.zeros(len(relax.H0)),
                    np.zeros(len(relax.H0)) ]
    relax.masks[0][:-9] = 1.0
    relax.masks[1][-9:] = 1.0
    return relax

def CellBFGS(atoms, alpha=70, bulk_modulus=140, poisson_ratio=0.3):
    """
        Create advanced BFGS optimizer for bulk systems, with 

        Parameters:
           bulk_modulus  Bulk Modulus of the isotropic system used to set up
                         the Hessian (in GPa). Default value 500 GPa corresponds
                         approximately to diamond.

           poisson_ratio Poisson ratio of the isotropic system ysed to set up
                         the initial Hessian (unitless, between -1 and 0.5).
                         Default value is 0.3.

           alpha         Diagonal elements of initial BFGS Hessian for atomic
                         coordinates (in eV/Å^2). Default value of 70
                         approximately. alpha is also used to suppress
                         collective rotation of all unit cell vectors (the
                         energy is invariant to all orthonormal
                         transformations of the unit cell). 

    """
    assert np.all(atoms.pbc)
 
    bulk_modulus = 0.00624150907 * bulk_modulus # GPa to eV / Å^3
    atoms_and_cell = ExpCellFilter(atoms)
    relax = BFGS(atoms_and_cell, alpha=alpha, logfile='/dev/null')
    g = np.eye(3)

    # https://scienceworld.wolfram.com/physics/LameConstants.html
    _lambda = 3 * bulk_modulus * (poisson_ratio) / (1 + 1 * poisson_ratio)
    _mu = _lambda * (1 - 2 * poisson_ratio) / (2 * poisson_ratio)

    # https://en.wikipedia.org/wiki/Elasticity_tensor
    g_ij = np.eye(3)

    # Construct 4th rank Elasticity tensor for isotropic systems
    C_ijkl = _lambda * np.einsum('ij,kl->ijkl',g_ij,g_ij)
    C_ijkl += _mu * (np.einsum('ik,jl->ijkl', g_ij, g_ij) +
                     np.einsum('il,kj->ijkl', g_ij, g_ij))

    # Supplement the tensor with suppression of pure rotations (which are right now 0 eigenvalues)
    # Loop over all basis vectors of skew symmetric real matrix
    for i,j in ((0,1),(0,2),(1,2)):
        Q = np.zeros((3,3))
        Q[i,j], Q[j,i] = 1, -1
        C_ijkl += np.einsum('ij,kl->ijkl', Q, Q) * alpha / 2

    # Update the Hessian initial guess
    #relax.H0[-9:, -9:] = np.eye(9) * np.trace(C_ijkl.reshape((9,9)))/9 * atoms.cell.volume
    relax.H0[-9:, -9:] = C_ijkl.reshape((9,9)) * atoms.cell.volume
    print(relax.H0[-9:,-9:])
    relax.masks = [ np.zeros(len(relax.H0)),
                    np.zeros(len(relax.H0)) ]
    relax.masks[0][:-9] = 1.0
    relax.masks[1][-9:] = 1.0
    relax.masks = [ np.zeros(len(relax.H0)) ]
    relax.masks[0][:] = 1.0
             
    return relax

