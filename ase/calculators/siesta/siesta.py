"""
This module defines the ASE interface to SIESTA.

Written by Mads Engelund (see www.espeem.com)

Home of the SIESTA package:
http://www.uam.es/departamentos/ciencias/fismateriac/siesta

2017.04 - Pedro Brandimarte: changes for python 2-3 compatible

"""

from dataclasses import dataclass
import os
import re
import shutil
import tempfile
from typing import Any, Dict, List

import numpy as np

from ase import Atoms
from ase.calculators.calculator import (
    FileIOCalculator, Parameters, ReadError)
from ase.calculators.siesta.parameters import PAOBasisBlock, format_fdf
from ase.calculators.siesta.import_ion_xml import get_ion
from ase.data import atomic_numbers
from ase.io.siesta import read_siesta_xv
from ase.io.siesta_input import SiestaInput
from ase.units import Bohr, Ry, eV
from ase.utils import deprecated
from pathlib import Path

meV = 0.001 * eV


def parse_siesta_version(output: bytes) -> str:
    match = re.search(rb'Siesta Version\s*:\s*(\S+)', output)

    if match is None:
        raise RuntimeError('Could not get Siesta version info from output '
                           '{!r}'.format(output))

    string = match.group(1).decode('ascii')
    return string


def get_siesta_version(executable: str) -> str:
    """ Return SIESTA version number.

    Run the command, for instance 'siesta' and
    then parse the output in order find the
    version number.
    """
    # XXX We need a test of this kind of function.  But Siesta().command
    # is not enough to tell us how to run Siesta, because it could contain
    # all sorts of mpirun and other weird parts.

    temp_dirname = tempfile.mkdtemp(prefix='siesta-version-check-')
    try:
        from subprocess import PIPE, Popen
        proc = Popen([executable],
                     stdin=PIPE,
                     stdout=PIPE,
                     stderr=PIPE,
                     cwd=temp_dirname)
        output, _ = proc.communicate()
        # We are not providing any input, so Siesta will give us a failure
        # saying that it has no Chemical_species_label and exit status 1
        # (as of siesta-4.1-b4)
    finally:
        shutil.rmtree(temp_dirname)

    return parse_siesta_version(output)


def format_block(name, block):
    lines = [f'%block {name}']
    for row in block:
        data = ' '.join(str(obj) for obj in row)
        lines.append(f'    {data}')
    lines.append(f'%endblock {name}')
    return '\n'.join(lines)


def bandpath2bandpoints(path):
    return '\n'.join([
        'BandLinesScale ReciprocalLatticeVectors',
        format_block('BandPoints', path.kpts)])


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


class SiestaParameters(Parameters):
    def __init__(
            self,
            label='siesta',
            mesh_cutoff=200 * Ry,
            energy_shift=100 * meV,
            kpts=None,
            xc='LDA',
            basis_set='DZP',
            spin='non-polarized',
            species=(),
            pseudo_qualifier=None,
            pseudo_path=None,
            symlink_pseudos=None,
            atoms=None,
            restart=None,
            fdf_arguments=None,
            atomic_coord_format='xyz',
            bandpath=None):
        kwargs = locals()
        kwargs.pop('self')
        Parameters.__init__(self, **kwargs)


def _nonpolarized_alias(_: List, kwargs: Dict[str, Any]) -> bool:
    if kwargs.get("spin", None) == "UNPOLARIZED":
        kwargs["spin"] = "non-polarized"
        return True
    return False


class Siesta(FileIOCalculator):
    """Calculator interface to the SIESTA code.
    """
    allowed_xc = {
        'LDA': ['PZ', 'CA', 'PW92'],
        'GGA': ['PW91', 'PBE', 'revPBE', 'RPBE',
                'WC', 'AM05', 'PBEsol', 'PBEJsJrLO',
                'PBEGcGxLO', 'PBEGcGxHEG', 'BLYP'],
        'VDW': ['DRSLL', 'LMKLL', 'KBM', 'C09', 'BH', 'VV']}

    name = 'siesta'
    _legacy_default_command = 'siesta < PREFIX.fdf > PREFIX.out'
    implemented_properties = [
        'energy',
        'free_energy',
        'forces',
        'stress',
        'dipole',
        'eigenvalues',
        'density',
        'fermi_energy']

    # Dictionary of valid input vaiables.
    default_parameters = SiestaParameters()

    # XXX Not a ASE standard mechanism (yet).  We need to communicate to
    # ase.spectrum.band_structure.calculate_band_structure() that we expect
    # it to use the bandpath keyword.
    accepts_bandpath_keyword = True

    fileio_rules = FileIOCalculator.ruleset(
        stdin_name='{prefix}.fdf',
        stdout_name='{prefix}.out')

    def __init__(self, command=None, profile=None, directory='.', **kwargs):
        """ASE interface to the SIESTA code.

        Parameters:
           - label        : The basename of all files created during
                            calculation.
           - mesh_cutoff  : Energy in eV.
                            The mesh cutoff energy for determining number of
                            grid points in the matrix-element calculation.
           - energy_shift : Energy in eV
                            The confining energy of the basis set generation.
           - kpts         : Tuple of 3 integers, the k-points in different
                            directions.
           - xc           : The exchange-correlation potential. Can be set to
                            any allowed value for either the Siesta
                            XC.funtional or XC.authors keyword. Default "LDA"
           - basis_set    : "SZ"|"SZP"|"DZ"|"DZP"|"TZP", strings which specify
                            the type of functions basis set.
           - spin         : "non-polarized"|"collinear"|
                            "non-collinear|spin-orbit".
                            The level of spin description to be used.
           - species      : None|list of Species objects. The species objects
                            can be used to to specify the basis set,
                            pseudopotential and whether the species is ghost.
                            The tag on the atoms object and the element is used
                            together to identify the species.
           - pseudo_path  : None|path. This path is where
                            pseudopotentials are taken from.
                            If None is given, then then the path given
                            in $SIESTA_PP_PATH will be used.
           - pseudo_qualifier: None|string. This string will be added to the
                            pseudopotential path that will be retrieved.
                            For hydrogen with qualifier "abc" the
                            pseudopotential "H.abc.psf" will be retrieved.
           - symlink_pseudos: None|bool
                            If true, symlink pseudopotentials
                            into the calculation directory, else copy them.
                            Defaults to true on Unix and false on Windows.
           - atoms        : The Atoms object.
           - restart      : str.  Prefix for restart file.
                            May contain a directory.
                            Default is  None, don't restart.
           - fdf_arguments: Explicitly given fdf arguments. Dictonary using
                            Siesta keywords as given in the manual. List values
                            are written as fdf blocks with each element on a
                            separate line, while tuples will write each element
                            in a single line.  ASE units are assumed in the
                            input.
           - atomic_coord_format: "xyz"|"zmatrix", strings to switch between
                            the default way of entering the system's geometry
                            (via the block AtomicCoordinatesAndAtomicSpecies)
                            and a recent method via the block Zmatrix. The
                            block Zmatrix allows to specify basic geometry
                            constrains such as realized through the ASE classes
                            FixAtom, FixedLine and FixedPlane.
        """

        # Put in the default arguments.
        parameters = self.default_parameters.__class__(**kwargs)

        # Call the base class.
        FileIOCalculator.__init__(
            self,
            command=command,
            profile=profile,
            directory=directory,
            **parameters)

    def __getitem__(self, key):
        """Convenience method to retrieve a parameter as
        calculator[key] rather than calculator.parameters[key]

            Parameters:
                -key       : str, the name of the parameters to get.
        """
        return self.parameters[key]

    def species(self, atoms):
        """Find all relevant species depending on the atoms object and
        species input.

            Parameters :
                - atoms : An Atoms object.
        """
        return SiestaInput.get_species(
            atoms, list(self['species']), self['basis_set'])

    @deprecated(
        "The keyword 'UNPOLARIZED' has been deprecated,"
        "and replaced by 'non-polarized'",
        category=FutureWarning,
        callback=_nonpolarized_alias,
    )
    def set(self, **kwargs):
        """Set all parameters.

            Parameters:
                -kwargs  : Dictionary containing the keywords defined in
                           SiestaParameters.

        .. deprecated:: 3.18.2
            The keyword 'UNPOLARIZED' has been deprecated and replaced by
            'non-polarized'
        """

        # XXX Inserted these next few lines because set() would otherwise
        # discard all previously set keywords to their defaults!  --askhl
        current = self.parameters.copy()
        current.update(kwargs)
        kwargs = current

        # Find not allowed keys.
        default_keys = list(self.__class__.default_parameters)
        offending_keys = set(kwargs) - set(default_keys)
        if len(offending_keys) > 0:
            mess = "'set' does not take the keywords: %s "
            raise ValueError(mess % list(offending_keys))

        # Use the default parameters.
        parameters = self.__class__.default_parameters.copy()
        parameters.update(kwargs)
        kwargs = parameters

        # Check energy inputs.
        for arg in ['mesh_cutoff', 'energy_shift']:
            value = kwargs.get(arg)
            if value is None:
                continue
            if not (isinstance(value, (float, int)) and value > 0):
                mess = "'{}' must be a positive number(in eV), \
                    got '{}'".format(arg, value)
                raise ValueError(mess)

        # Check the functional input.
        xc = kwargs.get('xc', 'LDA')
        if isinstance(xc, (tuple, list)) and len(xc) == 2:
            functional, authors = xc
            if functional.lower() not in [k.lower() for k in self.allowed_xc]:
                mess = f"Unrecognized functional keyword: '{functional}'"
                raise ValueError(mess)

            lsauthorslower = [a.lower() for a in self.allowed_xc[functional]]
            if authors.lower() not in lsauthorslower:
                mess = "Unrecognized authors keyword for %s: '%s'"
                raise ValueError(mess % (functional, authors))

        elif xc in self.allowed_xc:
            functional = xc
            authors = self.allowed_xc[xc][0]
        else:
            found = False
            for key, value in self.allowed_xc.items():
                if xc in value:
                    found = True
                    functional = key
                    authors = xc
                    break

            if not found:
                raise ValueError(f"Unrecognized 'xc' keyword: '{xc}'")
        kwargs['xc'] = (functional, authors)

        # Check fdf_arguments.
        if kwargs['fdf_arguments'] is None:
            kwargs['fdf_arguments'] = {}

        if not isinstance(kwargs['fdf_arguments'], dict):
            raise TypeError("fdf_arguments must be a dictionary.")

        # Call baseclass.
        FileIOCalculator.set(self, **kwargs)

    def set_fdf_arguments(self, fdf_arguments):
        """ Set the fdf_arguments after the initialization of the
            calculator.
        """
        self.validate_fdf_arguments(fdf_arguments)
        FileIOCalculator.set(self, fdf_arguments=fdf_arguments)

    def validate_fdf_arguments(self, fdf_arguments):
        """ Raises error if the fdf_argument input is not a
            dictionary of allowed keys.
        """
        # None is valid
        if fdf_arguments is None:
            return

        # Type checking.
        if not isinstance(fdf_arguments, dict):
            raise TypeError("fdf_arguments must be a dictionary.")

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write input (fdf)-file.
        See calculator.py for further details.

        Parameters:
            - atoms        : The Atoms object to write.
            - properties   : The properties which should be calculated.
            - system_changes : List of properties changed since last run.
        """

        super().write_input(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes)

        filename = self.getpath(ext='fdf')

        more_fdf_args = {}

        # Use the saved density matrix if only 'cell' and 'positions'
        # have changed.
        if (system_changes is None or
            ('numbers' not in system_changes and
             'initial_magmoms' not in system_changes and
             'initial_charges' not in system_changes)):

            more_fdf_args['DM.UseSaveDM'] = True

        if 'density' in properties:
            more_fdf_args['SaveRho'] = True

        # Start writing the file.
        with open(filename, 'w') as fd:
            self._write_fdf(fd, atoms, more_fdf_args)

    def _write_fdf(self, fd, atoms, more_fdf_args):
        # Write system name and label.
        fd.write(format_fdf('SystemName', self.prefix))
        fd.write(format_fdf('SystemLabel', self.prefix))
        fd.write("\n")

        # Write explicitly given options first to
        # allow the user to override anything.
        fdf_arguments = self['fdf_arguments']
        keys = sorted(fdf_arguments.keys())
        for key in keys:
            fd.write(format_fdf(key, fdf_arguments[key]))

        # Force siesta to return error on no convergence.
        # as default consistent with ASE expectations.
        if 'SCFMustConverge' not in fdf_arguments.keys():
            fd.write(format_fdf('SCFMustConverge', True))
        fd.write("\n")

        # Write spin level.
        fd.write(format_fdf('Spin     ', self['spin']))
        # Spin backwards compatibility.
        if self['spin'] == 'collinear':
            fd.write(
                format_fdf(
                    'SpinPolarized',
                    (True,
                     "# Backwards compatibility.")))
        elif self['spin'] == 'non-collinear':
            fd.write(
                format_fdf(
                    'NonCollinearSpin',
                    (True,
                     "# Backwards compatibility.")))

        # Write functional.
        functional, authors = self['xc']
        fd.write(format_fdf('XC.functional', functional))
        fd.write(format_fdf('XC.authors', authors))
        fd.write("\n")

        # Write mesh cutoff and energy shift.
        fd.write(format_fdf('MeshCutoff',
                            (self['mesh_cutoff'], 'eV')))
        fd.write(format_fdf('PAO.EnergyShift',
                            (self['energy_shift'], 'eV')))
        fd.write("\n")

        self._write_species(fd, atoms)
        self._write_structure(fd, atoms)

        for key, value in more_fdf_args.items():
            fd.write(format_fdf(key, value))

        if self["kpts"] is not None:
            kpts = np.array(self['kpts'])
            SiestaInput.write_kpts(fd, kpts)

        if self['bandpath'] is not None:
            lines = bandpath2bandpoints(self['bandpath'])
            fd.write(lines)
            fd.write('\n')

    def _write_species(self, fd, atoms):
        species, _ = self.species(atoms)

        if self['pseudo_path'] is not None:
            pseudo_path = self['pseudo_path']
        elif 'SIESTA_PP_PATH' in self.cfg:
            pseudo_path = self.cfg['SIESTA_PP_PATH']
        else:
            mess = "Please set the environment variable 'SIESTA_PP_PATH'"
            raise Exception(mess)

        species_info = SpeciesInfo(
            atoms=atoms,
            pseudo_path=Path(pseudo_path),
            pseudo_qualifier=self.pseudo_qualifier(),
            species=species,
            target_directory=Path(self.directory).resolve())

        species_info.write(fd)
        species_info.link_pseudos_into_directory(
            symlink_pseudos=self['symlink_pseudos'])

    def read(self, filename):
        """Read structural parameters from file .XV file
           Read other results from other files
           filename : siesta.XV
        """

        fname = self.getpath(filename)
        if not fname.exists():
            raise ReadError(f"The restart file '{fname}' does not exist")
        with fname.open() as fd:
            self.atoms = read_siesta_xv(fd)
        self.read_results()

    def getpath(self, fname=None, ext=None):
        """ Returns the directory/fname string """
        if fname is None:
            fname = self.prefix
        if ext is not None:
            fname = f'{fname}.{ext}'
        return Path(self.directory) / fname

    def _write_structure(self, fd, atoms):
        """Translate the Atoms object to fdf-format.

        Parameters
        ----------
        fd : IO
            An open file object.
        atoms: Atoms
            An atoms object.
        """
        cell = atoms.cell
        fd.write('\n')

        if cell.rank in [1, 2]:
            raise ValueError('Expected 3D unit cell or no unit cell.  You may '
                             'wish to add vacuum along some directions.')

        if np.any(cell):
            fd.write(format_fdf('LatticeConstant', '1.0 Ang'))
            fd.write(format_block('LatticeVectors', cell))

        _, species_numbers = self.species(atoms)
        write_atomic_coordinates(
            fd, atoms, species_numbers,
            self.parameters["atomic_coord_format"].lower())

        # Write magnetic moments.
        magmoms = atoms.get_initial_magnetic_moments()

        # The DM.InitSpin block must be written to initialize to
        # no spin. SIESTA default is FM initialization, if the
        # block is not written, but  we must conform to the
        # atoms object.
        if magmoms is not None:
            if len(magmoms) == 0:
                fd.write('#Empty block forces ASE initialization.\n')

            fd.write('%block DM.InitSpin\n')
            if len(magmoms) != 0 and isinstance(magmoms[0], np.ndarray):
                for n, M in enumerate(magmoms):
                    if M[0] != 0:
                        fd.write(
                            '    %d %.14f %.14f %.14f \n' %
                            (n + 1, M[0], M[1], M[2]))
            elif len(magmoms) != 0 and isinstance(magmoms[0], float):
                for n, M in enumerate(magmoms):
                    if M != 0:
                        fd.write('    %d %.14f \n' % (n + 1, M))
            fd.write('%endblock DM.InitSpin\n')
            fd.write('\n')

    def pseudo_qualifier(self):
        """Get the extra string used in the middle of the pseudopotential.
        The retrieved pseudopotential for a specific element will be
        'H.xxx.psf' for the element 'H' with qualifier 'xxx'. If qualifier
        is set to None then the qualifier is set to functional name.
        """
        if self['pseudo_qualifier'] is None:
            return self['xc'][0].lower()
        else:
            return self['pseudo_qualifier']

    def read_results(self):
        """Read the results.
        """
        self.read_number_of_grid_points()
        self.read_energy()
        self.read_forces_stress()
        self.read_eigenvalues()
        self.read_kpoints()
        self.read_dipole()

        self.read_ion(self.atoms)

        self.read_bands()

    def read_bands(self):
        bandpath = self['bandpath']
        if bandpath is None:
            return

        if len(bandpath.kpts) < 1:
            return

        fname = self.getpath(ext='bands')
        with open(fname) as fd:
            kpts, energies, efermi = read_bands_file(fd)
        bs = resolve_band_structure(bandpath, kpts, energies, efermi)
        self.results['bandstructure'] = bs

    def band_structure(self):
        return self.results['bandstructure']

    def read_ion(self, atoms):
        """
        Read the ion.xml file of each specie
        """
        species, species_numbers = self.species(atoms)

        self.results['ion'] = {}
        for species_number, spec in enumerate(species, start=1):
            symbol = spec['symbol']
            atomic_number = atomic_numbers[symbol]

            if spec['pseudopotential'] is None:
                if self.pseudo_qualifier() == '':
                    label = symbol
                else:
                    label = f"{symbol}.{self.pseudo_qualifier()}"
                pseudopotential = self.getpath(label, 'psf')
            else:
                pseudopotential = Path(spec['pseudopotential'])
                label = pseudopotential.stem

            name = f"{label}.{species_number}"
            if spec['ghost']:
                name = f"{name}.ghost"
                atomic_number = -atomic_number

            label = name.rsplit('.', 2)[0]

            if label not in self.results['ion']:
                fname = self.getpath(label, 'ion.xml')
                fname = Path(fname)
                if fname.is_file():
                    self.results['ion'][label] = get_ion(fname)
                else:
                    fname = self.getpath(label, 'psml')
                    fname = Path(fname)
                    if fname.is_file():
                        self.results['ion'][label] = get_ion(fname)

    def read_number_of_grid_points(self):
        """Read number of grid points from SIESTA's text-output file. """

        fname = self.getpath(ext='out')
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
        fname = self.getpath(ext='out')
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
        fname = self.getpath('FORCE_STRESS')
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

        file_name = self.getpath(ext='EIG')
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

        fname = self.getpath(ext='KP')
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

        self.results['kpoints'] = kpoints
        self.results['kweights'] = kweights

    def read_dipole(self):
        """Read dipole moment. """
        dipole = np.zeros([1, 3])
        with open(self.getpath(ext='out')) as fd:
            for line in fd:
                if line.rfind('Electric dipole (Debye)') > -1:
                    dipole = np.array([float(f) for f in line.split()[5:8]])
        # debye to e*Ang
        self.results['dipole'] = dipole * 0.2081943482534

    def get_fermi_level(self):
        return self.results['fermi_energy']

    def get_k_point_weights(self):
        return self.results['kweights']

    def get_ibz_k_points(self):
        return self.results['kpoints']


def write_atomic_coordinates(fd, atoms: Atoms, species_numbers,
                             atomic_coord_format: str):
    """Write atomic coordinates.

    Parameters
    ----------
    fd : IO
        An open file object.
    atoms : Atoms
        An atoms object.
    """
    if atomic_coord_format == 'xyz':
        write_atomic_coordinates_xyz(fd, atoms, species_numbers)
    elif atomic_coord_format == 'zmatrix':
        write_atomic_coordinates_zmatrix(fd, atoms, species_numbers)
    else:
        raise RuntimeError(
            f'Unknown atomic_coord_format: {atomic_coord_format}')


def write_atomic_coordinates_zmatrix(fd, atoms: Atoms, species_numbers):
    """Write atomic coordinates in Z-matrix format.

    Parameters
    ----------
    fd : IO
        An open file object.
    atoms : Atoms
        An atoms object.
    """
    fd.write('\n')
    fd.write('ZM.UnitsLength   Ang\n')
    fd.write('%block Zmatrix\n')
    fd.write('  cartesian\n')
    fstr = "{:5d}" + "{:20.10f}" * 3 + "{:3d}" * 3 + "{:7d} {:s}\n"
    a2constr = SiestaInput.make_xyz_constraints(atoms)
    a2p, a2s = atoms.get_positions(), atoms.get_chemical_symbols()
    for ia, (sp, xyz, ccc, sym) in enumerate(zip(species_numbers,
                                                 a2p,
                                                 a2constr,
                                                 a2s)):
        fd.write(fstr.format(
            sp, xyz[0], xyz[1], xyz[2], ccc[0],
            ccc[1], ccc[2], ia + 1, sym))
    fd.write('%endblock Zmatrix\n')

    # origin = tuple(-atoms.get_celldisp().flatten())
    # fd.write(format_block('AtomicCoordinatesOrigin', [origin]))


def write_atomic_coordinates_xyz(fd, atoms: Atoms, species_numbers):
    """Write atomic coordinates.

    Parameters
    ----------
    fd : IO
        An open file object.
    atoms : Atoms
        An atoms object.
    """
    fd.write('\n')
    fd.write('AtomicCoordinatesFormat  Ang\n')
    fd.write(format_block('AtomicCoordinatesAndAtomicSpecies',
                          [[*atom.position, number]
                           for atom, number in zip(atoms, species_numbers)]))
    fd.write('\n')

    # origin = tuple(-atoms.get_celldisp().flatten())
    # fd.write(format_block('AtomicCoordinatesOrigin', [origin]))


@dataclass
class SpeciesInfo:
    atoms: Atoms
    pseudo_path: Path
    pseudo_qualifier: str
    species: dict  # actually a kind of Parameters object, should refactor
    target_directory: Path

    def __post_init__(self):
        pao_basis = []
        chemical_labels = []
        basis_sizes = []
        file_instructions = []

        for species_number, spec in enumerate(self.species, start=1):
            symbol = spec['symbol']
            atomic_number = atomic_numbers[symbol]

            if spec['pseudopotential'] is None:
                if self.pseudo_qualifier == '':
                    label = symbol
                else:
                    label = f"{symbol}.{self.pseudo_qualifier}"
                src_path = self.pseudo_path / f"{label}.psf"
            else:
                src_path = Path(spec['pseudopotential'])
                label = src_path.stem
            if not src_path.is_absolute():
                src_path = self.pseudo_path / src_path
            if not src_path.exists():
                src_path = self.pseudo_path / f"{symbol}.psml"

            name = src_path.name
            name = name.split('.')
            name.insert(-1, str(species_number))
            if spec['ghost']:
                name.insert(-1, 'ghost')
                atomic_number = -atomic_number

            name = '.'.join(name)
            dst_path = self.target_directory / name

            instr = FileInstruction(src_path, dst_path)
            file_instructions.append(instr)

            label = '.'.join(np.array(name.split('.'))[:-1])
            string = '    %d %d %s' % (species_number, atomic_number, label)
            chemical_labels.append(string)
            if isinstance(spec['basis_set'], PAOBasisBlock):
                pao_basis.append(spec['basis_set'].script(label))
            else:
                basis_sizes.append(("    " + label, spec['basis_set']))

        self.file_instructions = file_instructions
        self.chemical_labels = chemical_labels
        self.pao_basis = pao_basis
        self.basis_sizes = basis_sizes

    def link_pseudos_into_directory(self, symlink_pseudos=None):
        if symlink_pseudos is None:
            symlink_pseudos = os.name != 'nt'

        for instruction in self.file_instructions:
            if symlink_pseudos:
                instruction.symlink()
            else:
                instruction.copyfile()

    def write(self, fd):
        fd.write(format_fdf('NumberOfSpecies', len(self.species)))
        fd.write(format_fdf('NumberOfAtoms', len(self.atoms)))

        fd.write(format_fdf('ChemicalSpecieslabel', self.chemical_labels))
        fd.write('\n')
        fd.write(format_fdf('PAO.Basis', self.pao_basis))
        fd.write(format_fdf('PAO.BasisSizes', self.basis_sizes))
        fd.write('\n')


@dataclass
class FileInstruction:
    src_path: Path
    dst_path: Path

    def copyfile(self):
        self._link(shutil.copy)

    def symlink(self):
        self._link(os.symlink)

    def _link(self, file_operation):
        if self.src_path == self.dst_path:
            return

        self.dst_path.unlink(missing_ok=True)
        file_operation(self.src_path, self.dst_path)
