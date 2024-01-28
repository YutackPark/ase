"""Effective medium theory potential."""

from math import sqrt

import numpy as np

from ase.calculators.calculator import (Calculator,
                                        PropertyNotImplementedError,
                                        all_changes)
from ase.data import atomic_numbers, chemical_symbols
from ase.neighborlist import NeighborList
from ase.units import Bohr

parameters = {
    #      E0     s0    V0     eta2    kappa   lambda  n0
    #      eV     bohr  eV     bohr^-1 bohr^-1 bohr^-1 bohr^-3
    'Al': (-3.28, 3.00, 1.493, 1.240, 2.000, 1.169, 0.00700),
    'Cu': (-3.51, 2.67, 2.476, 1.652, 2.740, 1.906, 0.00910),
    'Ag': (-2.96, 3.01, 2.132, 1.652, 2.790, 1.892, 0.00547),
    'Au': (-3.80, 3.00, 2.321, 1.674, 2.873, 2.182, 0.00703),
    'Ni': (-4.44, 2.60, 3.673, 1.669, 2.757, 1.948, 0.01030),
    'Pd': (-3.90, 2.87, 2.773, 1.818, 3.107, 2.155, 0.00688),
    'Pt': (-5.85, 2.90, 4.067, 1.812, 3.145, 2.192, 0.00802),
    # extra parameters - just for fun ...
    'H': (-3.21, 1.31, 0.132, 2.652, 2.790, 3.892, 0.00547),
    'C': (-3.50, 1.81, 0.332, 1.652, 2.790, 1.892, 0.01322),
    'N': (-5.10, 1.88, 0.132, 1.652, 2.790, 1.892, 0.01222),
    'O': (-4.60, 1.95, 0.332, 1.652, 2.790, 1.892, 0.00850)}

beta = 1.809  # (16 * pi / 3)**(1.0 / 3) / 2**0.5, preserve historical rounding


class EMT(Calculator):
    """Python implementation of the Effective Medium Potential.

    Supports the following standard EMT metals:
    Al, Cu, Ag, Au, Ni, Pd and Pt.

    In addition, the following elements are supported.
    They are NOT well described by EMT, and the parameters
    are not for any serious use:
    H, C, N, O

    The potential takes a single argument, ``asap_cutoff``
    (default: False).  If set to True, the cutoff mimics
    how Asap does it; most importantly the global cutoff
    is chosen from the largest atom present in the simulation,
    if False it is chosen from the largest atom in the parameter
    table.  True gives the behaviour of the Asap code and
    older EMT implementations, although the results are not
    bitwise identical.
    """
    implemented_properties = ['energy', 'free_energy', 'energies', 'forces',
                              'stress', 'magmom', 'magmoms']

    nolabel = True

    default_parameters = {'asap_cutoff': False}

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def initialize(self, atoms):
        self.rc = 0.0
        numbers = atoms.get_atomic_numbers()
        if self.parameters['asap_cutoff']:
            relevant_pars = {}
            for symb, p in parameters.items():
                if atomic_numbers[symb] in numbers:
                    relevant_pars[symb] = p
        else:
            relevant_pars = parameters
        maxseq = max(par[1] for par in relevant_pars.values()) * Bohr
        rc = self.rc = beta * maxseq * 0.5 * (sqrt(3) + sqrt(4))
        rr = rc * 2 * sqrt(4) / (sqrt(3) + sqrt(4))
        self.acut = np.log(9999.0) / (rr - rc)
        if self.parameters['asap_cutoff']:
            self.rc_list = self.rc * 1.045
        else:
            self.rc_list = self.rc + 0.5

        self.par = {}
        for Z in numbers:
            if Z in self.par:
                continue  # already stored
            sym = chemical_symbols[Z]
            if sym not in parameters:
                raise NotImplementedError(f'No EMT-potential for {sym}')
            p = parameters[sym]
            s0 = p[1] * Bohr
            eta2 = p[3] / Bohr
            kappa = p[4] / Bohr
            gamma1, gamma2 = self._calc_gammas(s0, eta2, kappa)
            self.par[Z] = {
                'E0': p[0],
                's0': s0,
                'V0': p[2],
                'eta2': eta2,
                'kappa': kappa,
                'lambda': p[5] / Bohr,
                'n0': p[6] / Bohr**3,
                'rc': rc,
                'gamma1': gamma1,
                'gamma2': gamma2,
            }

        self.chi = {}
        for s1, p1 in self.par.items():
            self.chi[s1] = {}
            for s2, p2 in self.par.items():
                self.chi[s1][s2] = p2['n0'] / p1['n0']

        self.energies = np.empty(len(atoms))
        self.forces = np.empty((len(atoms), 3))
        self.stress = np.empty((3, 3))
        self.deds = np.empty(len(atoms))

        self.nl = NeighborList([0.5 * self.rc_list] * len(atoms),
                               self_interaction=False, bothways=True)

    def _calc_gammas(self, s0, eta2, kappa):
        n = np.array([12, 6, 24])  # numbers of 1, 2, 3NN sites in fcc
        r = beta * s0 * np.sqrt([1.0, 2.0, 3.0])  # distances of 1, 2, 3NNs
        w = 1.0 / (1.0 + np.exp(self.acut * (r - self.rc)))
        x = n * w / 12.0
        gamma1 = x @ np.exp(-eta2 * (r - beta * s0))
        gamma2 = x @ np.exp(-kappa / beta * (r - beta * s0))
        return gamma1, gamma2

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if 'numbers' in system_changes:
            self.initialize(self.atoms)

        self.nl.update(self.atoms)

        self.energies[:] = 0.0
        self.forces[:] = 0.0
        self.stress[:] = 0.0
        self.deds[:] = 0.0

        natoms = len(self.atoms)

        for a1 in range(natoms):
            neighbors, d, r = self._get_neighbors(a1)
            if len(neighbors) == 0:
                continue

            p1, p2 = self._set_parameters(a1, neighbors, d, r)

            self._calc_theta(p2)
            self._calc_dsigma1(p1, p2)
            self._calc_dsigma2(p1, p2)

            self._calc_energies_c_as2(a1, p1, p2)
            self._calc_energies_forces_as1(a1, p1, p2)

        for a1 in range(natoms):
            neighbors, d, r = self._get_neighbors(a1)
            if len(neighbors) == 0:
                continue

            p1, p2 = self._set_parameters(a1, neighbors, d, r)

            self._calc_theta(p2)
            self._calc_dsigma1(p1, p2)

            self._calc_forces_c_as2(a1, neighbors, p1, p2)

        # subtract E0 (ASAP convention)
        self.energies -= [self.par[_]['E0'] for _ in self.atoms.numbers]

        energy = self.energies.sum()
        self.results['energy'] = self.results['free_energy'] = energy
        self.results['energies'] = self.energies
        self.results['forces'] = self.forces

        if self.atoms.cell.rank == 3:
            self.stress += self.stress.T.copy()
            self.stress *= -0.5 / self.atoms.get_volume()
            self.results['stress'] = self.stress.flat[[0, 4, 8, 5, 2, 1]]
        elif 'stress' in properties:
            raise PropertyNotImplementedError

    def _get_neighbors(self, a1):
        positions = self.atoms.positions
        cell = self.atoms.cell
        neighbors, offsets = self.nl.get_neighbors(a1)
        offsets = np.dot(offsets, cell)
        d = positions[neighbors] + offsets - positions[a1]
        r = np.linalg.norm(d, axis=1)
        mask = r < self.rc_list
        return neighbors[mask], d[mask], r[mask]

    def _set_parameters(self, a1, a2, d, r):
        ks = ['E0', 's0', 'V0', 'eta2', 'lambda', 'kappa', 'gamma1', 'gamma2']
        p1 = {k: self.par[k][self.i2i[a1]] for k in ks}
        p2 = {k: self.par[k][self.i2i[a2]] for k in ks}
        p2['chi'] = self.chi[self.i2i[a1], self.i2i[a2]]
        p2['d'] = d
        p2['r'] = r
        p2['invr'] = 1.0 / r
        return p1, p2

    def _calc_theta(self, p2):
        """Calculate cutoff function and its r derivative"""
        p2['w'] = 1.0 / (1.0 + np.exp(self.acut * (p2['r'] - self.rc)))
        p2['dwdr_over_w'] = -1.0 * self.acut * (1.0 - p2['w'])

    def _calc_dsigma1(self, p1, p2):
        """Calculate contributions of neighbors to sigma1"""
        chi = p2['chi']
        w = p2['w']
        dr1 = p2['r'] - beta * p1['s0']
        dr2 = p2['r'] - beta * p2['s0']
        p2['dsigma1s'] = np.exp(-p2['eta2'] * dr2) * chi * w / p1['gamma1']
        p2['dsigma1o'] = np.exp(-p1['eta2'] * dr1) / chi * w / p2['gamma1']

    def _calc_dsigma2(self, p1, p2):
        """Calculate contributions of neighbors to sigma2"""
        chi = p2['chi']
        w = p2['w']
        ds1 = p2['r'] / beta - p1['s0']
        ds2 = p2['r'] / beta - p2['s0']
        p2['dsigma2s'] = np.exp(-p2['kappa'] * ds2) * chi * w / p1['gamma2']
        p2['dsigma2o'] = np.exp(-p1['kappa'] * ds1) / chi * w / p2['gamma2']

    def _calc_energies_c_as2(self, a1, p1, p2):
        """Calculate E_c and the second term of E_AS and their s derivatives"""
        sigma1 = p2['dsigma1s'].sum()
        ds = -1.0 * np.log(sigma1 / 12.0) / (beta * p1['eta2'])

        lmdds = p1['lambda'] * ds
        expneglmdds = np.exp(-1.0 * lmdds)
        self.energies[a1] += p1['E0'] * (1.0 + lmdds) * expneglmdds
        self.deds[a1] += -1.0 * p1['E0'] * p1['lambda'] * lmdds * expneglmdds

        sixv0expnegkppds = 6.0 * p1['V0'] * np.exp(-1.0 * p1['kappa'] * ds)
        self.energies[a1] += sixv0expnegkppds
        self.deds[a1] += -1.0 * p1['kappa'] * sixv0expnegkppds

        self.deds[a1] /= sigma1 * beta * p1['eta2']

    def _calc_energies_forces_as1(self, a1, p1, p2):
        """Calculate the first term of E_AS and derivatives"""
        e_self = -0.5 * p1['V0'] * p2['dsigma2s']
        e_othr = -0.5 * p2['V0'] * p2['dsigma2o']
        self.energies[a1] += 0.5 * (e_self + e_othr).sum(axis=0)

        d = p2['d']
        dwdr_over_w = p2['dwdr_over_w']
        tmp_self = e_self * (dwdr_over_w - p2['kappa'] / beta)
        tmp_othr = e_othr * (dwdr_over_w - p1['kappa'] / beta)
        f = ((tmp_self + tmp_othr) * p2['invr'])[:, None] * d
        self.forces[a1] += f.sum(axis=0)
        self.stress -= 0.5 * np.dot(f.T, d)

    def _calc_forces_c_as2(self, a1, a2, p1, p2):
        """Calculate forces from E_c and the second term of E_AS"""
        d = p2['d']
        dwdr_over_w = p2['dwdr_over_w']
        ddsigma1sdr = p2['dsigma1s'] * (dwdr_over_w - p2['eta2'])
        ddsigma1odr = p2['dsigma1o'] * (dwdr_over_w - p1['eta2'])
        tmp_self = -1.0 * self.deds[a1] * ddsigma1sdr
        tmp_othr = -1.0 * self.deds[a2] * ddsigma1odr
        f = ((tmp_self + tmp_othr) * p2['invr'])[:, None] * d
        self.forces[a1] += f.sum(axis=0)
        self.stress -= 0.5 * np.dot(f.T, d)


def main():
    import sys

    from ase.io import read, write
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    atoms = read(inputfile)
    atoms.calc = EMT()
    atoms.get_stress()
    write(outputfile, atoms)


if __name__ == '__main__':
    main()
