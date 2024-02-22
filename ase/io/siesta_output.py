from pathlib import Path

import numpy as np

from ase.calculators.siesta.import_ion_xml import get_ion
from ase.units import Bohr, Ry
from ase.data import atomic_numbers


class OutputReader:
    def __init__(self, prefix, directory, bandpath=None):
        self.prefix = prefix
        self.directory = directory
        self.bandpath = bandpath
        self.results = {}

    def read_results(self):
        self.read_number_of_grid_points()
        self.read_energy()
        self.read_forces_stress()
        self.read_eigenvalues()
        self.read_kpoints()
        self.read_dipole()

        self.read_bands()
        return self.results

    def _prefixed(self, extension):
        return self.directory / f'{self.prefix}.{extension}'

    def read_bands(self):
        bandpath = self.bandpath
        if bandpath is None:
            return

        if len(bandpath.kpts) < 1:
            return

        fname = self._prefixed('bands')
        with open(fname) as fd:
            kpts, energies, efermi = read_bands_file(fd)
        bs = resolve_band_structure(bandpath, kpts, energies, efermi)
        self.results['bandstructure'] = bs

    def read_number_of_grid_points(self):
        """Read number of grid points from SIESTA's text-output file. """

        fname = self.directory / f'{self.prefix}.out'
        with open(fname) as fd:
            for line in fd:
                line = line.strip().lower()
                if line.startswith('initmesh: mesh ='):
                    n_points = [int(word) for word in line.split()[3:8:2]]
                    self.results['n_grid_point'] = n_points
                    break
            else:
                raise RuntimeError

    def read_energy(self):
        """Read energy from SIESTA's text-output file.
        """
        fname = self._prefixed('out')
        with open(fname) as fd:
            text = fd.read().lower()

        assert 'final energy' in text
        lines = iter(text.split('\n'))

        # Get the energy and free energy the last time it appears
        for line in lines:
            has_energy = line.startswith('siesta: etot    =')
            if has_energy:
                self.results['energy'] = float(line.split()[-1])
                line = next(lines)
                self.results['free_energy'] = float(line.split()[-1])

        if ('energy' not in self.results or
                'free_energy' not in self.results):
            raise RuntimeError

    def read_forces_stress(self):
        """Read the forces and stress from the FORCE_STRESS file.
        """
        fname = self.directory / 'FORCE_STRESS'
        with open(fname) as fd:
            lines = fd.readlines()

        stress_lines = lines[1:4]
        stress = np.empty((3, 3))
        for i in range(3):
            line = stress_lines[i].strip().split(' ')
            line = [s for s in line if len(s) > 0]
            stress[i] = [float(s) for s in line]

        self.results['stress'] = np.array(
            [stress[0, 0], stress[1, 1], stress[2, 2],
             stress[1, 2], stress[0, 2], stress[0, 1]])

        self.results['stress'] *= Ry / Bohr**3

        start = 5
        self.results['forces'] = np.zeros((len(lines) - start, 3), float)
        for i in range(start, len(lines)):
            line = [s for s in lines[i].strip().split(' ') if len(s) > 0]
            self.results['forces'][i - start] = [float(s) for s in line[2:5]]

        self.results['forces'] *= Ry / Bohr

    def read_eigenvalues(self):
        """ A robust procedure using the suggestion by Federico Marchesin """

        file_name = self._prefixed('EIG')
        try:
            with open(file_name) as fd:
                self.results['fermi_energy'] = float(fd.readline())
                n, num_hamilton_dim, nkp = map(int, fd.readline().split())
                _ee = np.split(
                    np.array(fd.read().split()).astype(float), nkp)
        except OSError:
            return 1

        n_spin = 1 if num_hamilton_dim > 2 else num_hamilton_dim
        ksn2e = np.delete(_ee, 0, 1).reshape([nkp, n_spin, n])

        eig_array = np.empty((n_spin, nkp, n))
        eig_array[:] = np.inf

        for k, sn2e in enumerate(ksn2e):
            for s, n2e in enumerate(sn2e):
                eig_array[s, k, :] = n2e

        assert np.isfinite(eig_array).all()

        self.results['eigenvalues'] = eig_array
        return 0

    def read_kpoints(self):
        """ Reader of the .KP files """

        fname = self._prefixed('KP')
        try:
            with open(fname) as fd:
                nkp = int(next(fd))
                kpoints = np.empty((nkp, 3))
                kweights = np.empty(nkp)

                for i in range(nkp):
                    line = next(fd)
                    tokens = line.split()
                    numbers = np.array(tokens[1:]).astype(float)
                    kpoints[i] = numbers[:3]
                    kweights[i] = numbers[3]

        except (OSError):
            return 1

        self.results['kpoints'] = kpoints
        self.results['kweights'] = kweights

        return 0

    def read_dipole(self):
        """Read dipole moment. """
        dipole = np.zeros([1, 3])
        with open(self._prefixed('out')) as fd:
            for line in fd:
                if line.rfind('Electric dipole (Debye)') > -1:
                    dipole = np.array([float(f) for f in line.split()[5:8]])
        # debye to e*Ang
        self.results['dipole'] = dipole * 0.2081943482534


def read_bands_file(fd):
    efermi = float(next(fd))
    next(fd)  # Appears to be max/min energy.  Not important for us
    header = next(fd)  # Array shape: nbands, nspins, nkpoints
    nbands, nspins, nkpts = np.array(header.split()).astype(int)

    # three fields for kpt coords, then all the energies
    ntokens = nbands * nspins + 3

    # Read energies for each kpoint:
    data = []
    for i in range(nkpts):
        line = next(fd)
        tokens = line.split()
        while len(tokens) < ntokens:
            # Multirow table.  Keep adding lines until the table ends,
            # which should happen exactly when we have all the energies
            # for this kpoint.
            line = next(fd)
            tokens += line.split()
        assert len(tokens) == ntokens
        values = np.array(tokens).astype(float)
        data.append(values)

    data = np.array(data)
    assert len(data) == nkpts
    kpts = data[:, :3]
    energies = data[:, 3:]
    energies = energies.reshape(nkpts, nspins, nbands)
    assert energies.shape == (nkpts, nspins, nbands)
    return kpts, energies, efermi


def resolve_band_structure(path, kpts, energies, efermi):
    """Convert input BandPath along with Siesta outputs into BS object."""
    # Right now this function doesn't do much.
    #
    # Not sure how the output kpoints in the siesta.bands file are derived.
    # They appear to be related to the lattice parameter.
    #
    # We should verify that they are consistent with our input path,
    # but since their meaning is unclear, we can't quite do so.
    #
    # Also we should perhaps verify the cell.  If we had the cell, we
    # could construct the bandpath from scratch (i.e., pure outputs).
    from ase.spectrum.band_structure import BandStructure
    ksn2e = energies
    skn2e = np.swapaxes(ksn2e, 0, 1)
    bs = BandStructure(path, skn2e, reference=efermi)
    return bs
