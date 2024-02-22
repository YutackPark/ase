"""
This module defines the ASE interface to SIESTA.

Written by Mads Engelund (see www.espeem.com)

Home of the SIESTA package:
http://www.uam.es/departamentos/ciencias/fismateriac/siesta

2017.04 - Pedro Brandimarte: changes for python 2-3 compatible

"""

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
from ase.data import atomic_numbers
from ase.io.siesta import read_siesta_xv
from ase.io.siesta_input import SiestaInput
from ase.units import Ry, eV
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


def bandpath2bandpoints(path):
    lines = []
    add = lines.append

    add('BandLinesScale ReciprocalLatticeVectors\n')
    add('%block BandPoints\n')
    for kpt in path.kpts:
        add('    {:18.15f} {:18.15f} {:18.15f}\n'.format(*kpt))
    add('%endblock BandPoints')
    return ''.join(lines)


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
        # Call base calculator.
        FileIOCalculator.write_input(
            self,
            atoms=atoms,
            properties=properties,
            system_changes=system_changes)

        if system_changes is None and properties is None:
            return

        filename = self.getpath(ext='fdf')

        # On any changes, remove all analysis files.
        if system_changes is not None:
            self.remove_analysis()

        # Start writing the file.
        with open(filename, 'w') as fd:
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

            # Write the minimal arg
            self._write_species(fd, atoms)
            self._write_structure(fd, atoms)

            # Use the saved density matrix if only 'cell' and 'positions'
            # have changed.
            if (system_changes is None or
                ('numbers' not in system_changes and
                 'initial_magmoms' not in system_changes and
                 'initial_charges' not in system_changes)):
                fd.write(format_fdf('DM.UseSaveDM', True))

            # Save density.
            if 'density' in properties:
                fd.write(format_fdf('SaveRho', True))

            if self["kpts"] is not None:
                kpts = np.array(self['kpts'])
                SiestaInput.write_kpts(fd, kpts)

            if self['bandpath'] is not None:
                lines = bandpath2bandpoints(self['bandpath'])
                fd.write(lines)
                fd.write('\n')

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

    def remove_analysis(self):
        """ Remove all analysis files"""
        filename = Path(self.getpath(ext='RHO'))
        if filename.exists():
            filename.unlink()

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

        # Write lattice vectors
        if np.any(cell):
            fd.write(format_fdf('LatticeConstant', '1.0 Ang'))
            fd.write('%block LatticeVectors\n')
            for i in range(3):
                for j in range(3):
                    s = ('    %.15f' % cell[i, j]).rjust(16) + ' '
                    fd.write(s)
                fd.write('\n')
            fd.write('%endblock LatticeVectors\n')
            fd.write('\n')

        self._write_atomic_coordinates(fd, atoms)

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

    def _write_atomic_coordinates(self, fd, atoms: Atoms):
        """Write atomic coordinates.

        Parameters
        ----------
        fd : IO
            An open file object.
        atoms : Atoms
            An atoms object.
        """
        af = self.parameters["atomic_coord_format"].lower()
        species, species_numbers = self.species(atoms)
        if af == 'xyz':
            self._write_atomic_coordinates_xyz(fd, atoms, species_numbers)
        elif af == 'zmatrix':
            self._write_atomic_coordinates_zmatrix(fd, atoms, species_numbers)
        else:
            raise RuntimeError(f'Unknown atomic_coord_format: {af}')

    def _write_atomic_coordinates_xyz(self, fd, atoms: Atoms, species_numbers):
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
        fd.write('%block AtomicCoordinatesAndAtomicSpecies\n')
        for atom, number in zip(atoms, species_numbers):
            xyz = atom.position
            line = ('    %.9f' % xyz[0]).rjust(16) + ' '
            line += ('    %.9f' % xyz[1]).rjust(16) + ' '
            line += ('    %.9f' % xyz[2]).rjust(16) + ' '
            line += str(number) + '\n'
            fd.write(line)
        fd.write('%endblock AtomicCoordinatesAndAtomicSpecies\n')
        fd.write('\n')

        origin = tuple(-atoms.get_celldisp().flatten())
        if any(origin):
            fd.write('%block AtomicCoordinatesOrigin\n')
            fd.write('     %.4f  %.4f  %.4f\n' % origin)
            fd.write('%endblock AtomicCoordinatesOrigin\n')
            fd.write('\n')

    def _write_atomic_coordinates_zmatrix(
            self, fd, atoms: Atoms, species_numbers):
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

        origin = tuple(-atoms.get_celldisp().flatten())
        if any(origin):
            fd.write('%block AtomicCoordinatesOrigin\n')
            fd.write('     %.4f  %.4f  %.4f\n' % origin)
            fd.write('%endblock AtomicCoordinatesOrigin\n')
            fd.write('\n')

    def _write_species(self, fd, atoms):
        """Write input related the different species.

        Parameters:
            - f:     An open file object.
            - atoms: An atoms object.
        """
        species, species_numbers = self.species(atoms)

        if self['pseudo_path'] is not None:
            pseudo_path = self['pseudo_path']
        elif 'SIESTA_PP_PATH' in self.cfg:
            pseudo_path = self.cfg['SIESTA_PP_PATH']
        else:
            mess = "Please set the environment variable 'SIESTA_PP_PATH'"
            raise Exception(mess)

        fd.write(format_fdf('NumberOfSpecies', len(species)))
        fd.write(format_fdf('NumberOfAtoms', len(atoms)))

        pao_basis = []
        chemical_labels = []
        basis_sizes = []
        synth_blocks = []

        pseudo_path = Path(pseudo_path)
        for species_number, spec in enumerate(species, start=1):
            symbol = spec['symbol']
            atomic_number = atomic_numbers[symbol]

            if spec['pseudopotential'] is None:
                if self.pseudo_qualifier() == '':
                    label = symbol
                else:
                    label = f"{symbol}.{self.pseudo_qualifier()}"
                pseudopotential = pseudo_path / f"{label}.psf"
            else:
                pseudopotential = Path(spec['pseudopotential'])
                label = pseudopotential.stem
            if not pseudopotential.is_absolute():
                pseudopotential = pseudo_path / pseudopotential
            if not pseudopotential.exists():
                label = symbol
                pseudopotential = pseudo_path / f"{label}.psml"
                if not pseudopotential.exists():
                    mess = f"Pseudopotential '{pseudopotential}' not found"
                    raise RuntimeError(mess)

            name = pseudopotential.name
            name = name.split('.')
            name.insert(-1, str(species_number))
            if spec['ghost']:
                name.insert(-1, 'ghost')
                atomic_number = -atomic_number

            name = '.'.join(name)
            pseudopath = self.getpath(name)
            if Path(os.getcwd()) / name != pseudopotential:
                if pseudopath.is_symlink() or pseudopath.is_file():
                    os.remove(pseudopath)
                symlink_pseudos = self['symlink_pseudos']

                if symlink_pseudos is None:
                    symlink_pseudos = not os.name == 'nt'

                if symlink_pseudos:
                    os.symlink(pseudopotential, pseudopath)
                else:
                    shutil.copy(pseudopotential, pseudopath)

            if len(synth_blocks) > 0:
                fd.write(format_fdf('SyntheticAtoms', list(synth_blocks)))

            label = '.'.join(np.array(name.split('.'))[:-1])
            string = '    %d %d %s' % (species_number, atomic_number, label)
            chemical_labels.append(string)
            if isinstance(spec['basis_set'], PAOBasisBlock):
                pao_basis.append(spec['basis_set'].script(label))
            else:
                basis_sizes.append(("    " + label, spec['basis_set']))
        fd.write(format_fdf('ChemicalSpecieslabel', chemical_labels))
        fd.write('\n')
        fd.write(format_fdf('PAO.Basis', pao_basis))
        fd.write(format_fdf('PAO.BasisSizes', basis_sizes))
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
        """Read the results."""
        from ase.io.siesta_output import OutputReader
        reader = OutputReader(prefix=self.prefix,
                              directory=Path(self.directory),
                              bandpath=self['bandpath'])
        results = reader.read_results()
        self.results.update(results)

        self.results['ion'] = self.read_ion(self.atoms)

    def read_ion(self, atoms):
        """
        Read the ion.xml file of each specie
        """
        species, species_numbers = self.species(atoms)

        ion_results = {}
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

            if label not in ion_results:
                fname = self.getpath(label, 'ion.xml')
                fname = Path(fname)
                if fname.is_file():
                    ion_results[label] = get_ion(fname)
                else:
                    fname = self.getpath(label, 'psml')
                    fname = Path(fname)
                    if fname.is_file():
                        ion_results[label] = get_ion(fname)

        return ion_results

    def band_structure(self):
        return self.results['bandstructure']

    def get_fermi_level(self):
        return self.results['fermi_energy']

    def get_k_point_weights(self):
        return self.results['kweights']

    def get_ibz_k_points(self):
        return self.results['kpoints']
