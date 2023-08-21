import warnings
from copy import deepcopy
from glob import glob
from os.path import basename, dirname, isfile
from pathlib import Path
from ase.cell import Cell
import time

import numpy as np

from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.units import Bohr

no_positions_error = (
    'no positions can be read from this onetep output '
    'if you wish to use ASE to read onetep outputs '
    'please use uppercase block positions in your calculations')

unable_to_read = (
    'unable to read this onetep output file, ending'
)

# taken from onetep source code,
# does not seem to be from any known NIST data
units = {
    'Hartree': 27.2116529,
    'Bohr': 1 / 1.889726134583548707935
}

# Want to add a functionality? add a global constant below
ONETEP_START = "Linear-Scaling Ab Initio Total Energy Program"
ONETEP_STOP = "Job completed:"
ONETEP_TOTAL_ENERGY = "*** NGWF optimisation converged ***"
ONETEP_FORCE = "* Forces *"
ONETEP_MULLIKEN = "Mulliken Atomic Populations"
ONETEP_IBFGS_ITER = "starting iteration"
ONETEP_POSITION = "Cell Contents"
ONETEP_FIRST_POSITION = "%BLOCK POSITIONS_"
ONETEP_WRONG_FIRST_POSITION = '%block positions_'
ONETEP_RESUMING_GEOM = 'Resuming previous ONETEP Geometry Optimisation'
ONETEP_CELL = "NOT IMPLEMENTED YET"
ONETEP_STRESS = "NOT IMPLEMENTED YET"
ONETEP_ATOM_COUNT = "Totals:"
ONETEP_IBFGS_IMPROVE = "improving iteration"
ONETEP_START_GEOM = "Starting ONETEP Geometry Optimisation"
ONETEP_SPECIES = "%BLOCK SPECIES "
ONETEP_SPECIESL = "%block species "
ONETEP_FIRST_CELLL = "%block lattice_cart"
ONETEP_FIRST_CELL = "%BLOCK LATTICE_CART"
ONETEP_STRESS_CELL = "stress_calculation: cell geometry"


# def _alphanum_sorted(arr):
"""
Sort an array of strings alphanumerically.
"""
#    def alphanum_key(key):
#        digits = re.split('([0-9]+)', key)
#        return [int(char) if char.isdigit() else char for char in digits]
#    return sorted(arr, key=alphanum_key)


def get_onetep_keywords(path):
    if not isfile(path):
        return None
    with open(path, 'r') as fd:
        results = next(read_onetep_in(fd))
    return results['keywords']


def read_onetep_in(fd):
    """
    Read a single ONETEP input.

    This function can be used to visually check ONETEP inputs,
    using the ase gui. It can also be used to get access to
    the input parameters attached to the ONETEP calculator
    returned

    The function should work on inputs which contain
    'include_file' command(s), (possibly recursively
    but untested)

    The function should work on input which contains
    exotic element(s) name(s) if the specie block is
    present to map them back to real element(s)

    Parameters
    ----------
    fd : io-object
        File to read.

    Yields
    ------
    structure: Atoms
        Atoms object with cell and a Onetep calculator
        attached which contains the keywords dictionary
    """

    fdi_lines = fd.readlines()

    def clean_lines(lines):
        """
        Remove indesirable line from the input
        """
        new_lines = []
        for line in lines:
            if line.startswith('#'):
                pass
            elif line.startswith('!'):
                pass
            elif line.strip():
                new_lines.append(line)
            else:
                pass
        return new_lines

    # Skip comments and empty lines
    fdi_lines = clean_lines(fdi_lines)

    # Are we in a block?
    in_block = 0

    keywords = {}
    atoms = Atoms()
    cell = np.zeros((3, 3))
    fractional = False
    positions = False
    symbols = False

    n = 0
    # Main loop reading the input
    while n < len(fdi_lines):
        line = fdi_lines[n]
        line_lower = line.lower()

        if '%block' in line_lower:
            in_block = n
            if 'lattice_cart' in line_lower:
                if 'ang' in fdi_lines[in_block + 1]:
                    cell = np.loadtxt(fdi_lines[n + 2:n + 5])
                else:
                    cell = np.loadtxt(fdi_lines[n + 1:n + 4])
                    cell *= Bohr

        if not in_block:
            if 'devel_code' in line_lower:
                warnings.warn('devel_code is not supported')
                n += 1
                continue
            sep = line.replace(':', ' ').replace('=', ' ').strip().split()
            keywords[sep[0]] = ' '.join(sep[1:])
            # If include_file is used, we open the included file
            # and insert it in the current fdi_lines...
            # Should work with a cascade
            if 'include_file' == sep[0]:
                new_fd = open(sep[1], 'r')
                new_lines = new_fd.readlines()
                new_lines = clean_lines(new_lines)
                fdi_lines = fdi_lines[:n] + \
                    new_lines + \
                    fdi_lines[n + 1:]
                continue

        if '%endblock' in line_lower:
            if 'positions_' in line_lower:
                if 'ang' in fdi_lines[in_block + 1]:
                    to_read = fdi_lines[in_block + 2:n]
                    positions = np.loadtxt(to_read, usecols=(1, 2, 3))
                else:
                    to_read = fdi_lines[in_block + 1:n]
                    positions = np.loadtxt(to_read, usecols=(1, 2, 3))
                    positions *= units['Bohr']
                symbols = np.loadtxt(to_read, usecols=(0), dtype='str')
                if 'frac' in line_lower:
                    fractional = True
            elif '%endblock species' == line_lower.strip():
                els = fdi_lines[in_block + 1:n]
                species = {}
                for el in els:
                    sep = el.split()
                    species[sep[0]] = sep[1]
                to_read = [i.strip() for i in fdi_lines[in_block + 1:n]]
                keywords['species'] = to_read
            elif 'lattice_cart' in line_lower:
                pass
            else:
                to_read = [i.strip() for i in fdi_lines[in_block + 1:n]]
                block_title = line_lower.replace('%endblock', '').strip()
                keywords[block_title] = to_read
            in_block = 0
        n += 1

    # Necessary if we have only one atom
    # Check if the cell is valid (3D)
    if not cell.any(axis=1).all():
        raise ValueError('invalid cell specified')

    if positions is False:
        raise ValueError('invalid position specified')

    if symbols is False:
        raise ValueError('no symbols found')

    positions = positions.reshape(-1, 3)
    symbols = symbols.reshape(-1)
    tags = []
    for symbol in symbols:
        label = symbol.replace(species[symbol], '')
        if label.isdigit():
            tags.append(int(label))
        else:
            tags.append(0)
    atoms = Atoms([species[i] for i in symbols])
    if fractional:
        # This is just dot product
        positions = positions @ cell
        positions /= units['Bohr']
    atoms.set_positions(positions)
    atoms.set_cell(cell)
    atoms.set_pbc(True)
    atoms.set_tags(tags)
    results = {'atoms': atoms, 'keywords': keywords}
    yield results


def write_onetep_in(
        fd,
        atoms,
        edft=False,
        xc='PBE',
        ngwf_count=-1,
        ngwf_radius=9.0,
        keywords={},
        pseudopotentials={},
        pseudo_path=".",
        pseudo_suffix=None,
        **kwargs):
    """
    Write a single ONETEP input.

    This function will be used by ASE to perform
    various workflows (Opt, NEB...) or can be used
    manually to quickly create ONETEP input file(s).

    The function will first write keywords in
    alphabetic order in lowercase. Secondly, blocks
    will be written in alphabetic order in uppercase.

    Two ways to work with the function:

        - By providing only (simple) keywords present in
        the parameters. ngwf_count and ngwf_radius
        accept multiple types as described in the Parameters
        section.

        - If the keywords parameters is provided as a dictionary
        these keywords will be used to write the input file and
        will take priority.

    If no pseudopotentials are provided in the parameters and
    the function will try to look for suitable pseudopotential
    in the pseudo_path.

    Parameters
    ----------
    fd : file
        File to write.
    atoms: Atoms
        Atoms including Cell object to write.
    edft: Bool
        Activate EDFT.
    xc: str
        DFT xc to use e.g (PBE, RPBE, ...)
    ngwf_count: int|list|dict
        Behaviour depends on the type:
            int: every species will have this amount
            of ngwfs.
            list: list of int, will be attributed
            alphabetically to species:
            dict: keys are species name(s),
            value are their number:
    ngwf_radius: int|list|dict
        Behaviour depends on the type:
            float: every species will have this radius.
            list: list of float, will be attributed
            alphabetically to species:
            [10.0, 9.0]
            dict: keys are species name(s),
            value are their radius:
            {'Na': 9.0, 'Cl': 10.0}
    keywords: dict
        Dictionary with ONETEP keywords to write,
        keywords with lists as values will be
        treated like blocks, with each element
        of list being a different line.
    pseudopotentials: dict
        Behaviour depends on the type:
            keys are species name(s) their
            value are the pseudopotential file to use:
            {'Na': 'Na.usp', 'Cl': 'Cl.usp'}
    pseudo_path: str
        Where to look for pseudopotential, correspond
        to the pseudo_path keyword of ONETEP.
    pseudo_suffix: str
        Suffix for the pseudopotential filename
        to look for, useful if you have multiple sets of
        pseudopotentials in pseudo_path.
    """

    label = kwargs.get('label', 'onetep')
    directory = kwargs.get('directory', Path(dirname(fd.name)))
    autorestart = kwargs.get('autorestart', False)
    elements = np.array(atoms.symbols)
    tags = np.array(atoms.get_tags())
    formatted_tags = np.array(['' if i == 0 else str(i) for i in tags])
    species = np.char.add(elements, formatted_tags)
    numbers = np.array(atoms.numbers)
    tmp = np.argsort(species)
    # We sort both Z and name the same
    numbers = np.take_along_axis(numbers, tmp, axis=0)
    # u_elements = np.take_along_axis(elements, tmp, axis=0)
    u_species = np.take_along_axis(species, tmp, axis=0)
    elements = np.take_along_axis(elements, tmp, axis=0)
    # We want to keep unique but without sort: small trick with index
    idx = np.unique(u_species, return_index=True)[1]
    elements = elements[idx]
    # Unique species
    u_species = u_species[idx]
    numbers = numbers[idx]

    n_sp = len(u_species)

    if isinstance(ngwf_count, int):
        ngwf_count = dict(zip(u_species, [ngwf_count] * n_sp))
    elif isinstance(ngwf_count, list):
        ngwf_count = dict(zip(u_species, ngwf_count))
    elif isinstance(ngwf_count, dict):
        pass
    else:
        raise TypeError('ngwf_count can only be int|list|dict')

    if isinstance(ngwf_radius, float):
        ngwf_radius = dict(zip(u_species, [ngwf_radius] * n_sp))
    elif isinstance(ngwf_radius, list):
        ngwf_radius = dict(zip(u_species, ngwf_radius))
    elif isinstance(ngwf_radius, dict):
        pass
    else:
        raise TypeError('ngwf_radius can only be float|list|dict')

    pp_files = keywords.get('pseudo_path', pseudo_path)
    pp_files = pp_files.replace('\'', '')
    pp_files = pp_files.replace('\"', '')
    pp_files = Path(pp_files).glob('*')
    pp_files = [i for i in sorted(pp_files) if i.is_file()]
    pp_is_manual = keywords.get('species_pot', False)
    common_suffix = ['.usp', '.recpot', '.upf', '.paw', '.psp', '.pspnc']
    if pseudo_suffix:
        common_suffix = [pseudo_suffix]
    # Transform to list
    
    if pp_is_manual:
        pp_list = keywords['species_pot']
    elif isinstance(pseudopotentials, dict):
        pp_list = []
        for idx, el in enumerate(u_species):
            try:
                pp_list.append(el + ' ' + pseudopotentials[el])
            except KeyError:
                for i in pp_files:
                    if elements[idx] in basename(i)[:2]:
                        for j in common_suffix:
                            if basename(i).endswith(j):
                                pp_list.append(el + ' ' + basename(i))
                #pp_maybe = attempt_to_find_pp(elements[idx])
                #if pp_maybe:
                #    pp_list.append(el + ' ' + pp_maybe)
                #else:
                #    warnings.warn('No pseudopotential found for element {}'
                #                  .format(el))
    else:
        raise TypeError('pseudopotentials object can only be dict')

    default_species = []
    for idx, el in enumerate(u_species):
        tmp = ""
        tmp += u_species[idx] + " " + elements[idx] + " "
        tmp += str(numbers[idx]) + " "
        try:
            tmp += str(ngwf_count[el]) + " "
        except KeyError:
            tmp += str(ngwf_count[elements[idx]]) + " "
        try:
            tmp += str(ngwf_radius[el])
        except KeyError:
            tmp += str(ngwf_radius[elements[idx]])
        default_species.append(tmp)

    positions_abs = ['ang']
    for s, p in zip(species, atoms.get_positions()):
        line = '{s:>5} {0:>12.6f} {1:>12.6f} {2:>12.6f}'.format(s=s, *p)
        positions_abs.append(line)

    lattice_cart = ['ang']
    for axis in atoms.get_cell():
        line = '{0:>16.8f} {1:>16.8f} {2:>16.8f}'.format(*axis)
        lattice_cart.append(line)

    # Default keywords if not provided by the user,
    # most of them are ONETEP default, except write_forces
    # which is always turned on.
    default_keywords = {
        "xc_functional": "pbe",
        "edft": edft,
        "cutoff_energy": 20,
        "paw": False,
        "task": "singlepoint",
        "output_detail": "normal",
        "species": default_species,
        "pseudo_path": pseudo_path,
        "species_pot": pp_list,
        "positions_abs": positions_abs,
        "lattice_cart": lattice_cart,
        "write_forces": True,
        "forces_output_detail": 'verbose'
    }

    # Main loop, fill the keyword dictionary
    keywords = {key.lower(): value for key, value in keywords.items()}
    for value in default_keywords:
        if not keywords.get(value, None):
            keywords[value] = default_keywords[value]

    # No pseudopotential provided, we look for them in pseudo_path
    # If autorestart is True, we look for restart files,
    # and turn on relevant keywords...
    if autorestart:
        keywords['read_denskern'] = \
            isfile(directory / (label + '.dkn'))
        keywords['read_tightbox_ngwfs'] = \
            isfile(directory / (label + '.tightbox_ngwfs'))
        keywords['read_hamiltonian'] = \
            isfile(directory / (label + '.ham'))

    # If not EDFT, hamiltonian is irrelevant.
    # print(keywords.get('edft', False))
    # keywords['read_hamiltonian'] = \
        # keywords.get('read_hamiltonian', False) & keywords.get('edft', False)

    keywords = dict(sorted(keywords.items()))

    lines = []
    block_lines = []

    for key, value in keywords.items():
        if isinstance(value, (list, np.ndarray)):
            if not all(isinstance(_, str) for _ in value):
                raise TypeError('list values for blocks must be strings only')
            block_lines.append(('\n%block ' + key).upper())
            block_lines.extend(value)
            block_lines.append(('%endblock ' + key).upper())
        elif isinstance(value, bool):
            lines.append(str(key) + " : " + str(value)[0])
        elif isinstance(value, (str, int, float)):
            lines.append(str(key) + " : " + str(value))
        else:
            raise TypeError('keyword values must be list|str|bool')
    fd.writelines(line + '\n' for line in lines)
    fd.writelines(b_line + '\n' for b_line in block_lines)

    if 'devel_code' in kwargs:
        warnings.warn('writing devel code as it is, at the end of the file')
        fd.writelines('\n' + line for line in kwargs['devel_code'])


def read_onetep_out(fd, index=-1, improving=False, **kwargs):
    """
    Read ONETEP output(s).

    !!!
    This function will be used by ASE when performing
    various workflows (Opt, NEB...)
    !!!

    Parameters
    ----------
    fd : file
        File to read.
    index: slice
        Which atomic configuration to read
    improving: Bool
        If the output is a geometry optimisation,
        improving = True will keep line search
        configuration from BFGS

    Yields
    ------
    structure: Atoms|list of Atoms
    """
    # Put everything in memory
    fdo_lines = fd.readlines()
    n_lines = len(fdo_lines)

    # Used to store index of important elements
    output = {
        ONETEP_START: [],
        ONETEP_STOP: [],
        ONETEP_TOTAL_ENERGY: [],
        ONETEP_FORCE: [],
        ONETEP_MULLIKEN: [],
        ONETEP_STRESS: [],
        ONETEP_POSITION: [],
        ONETEP_FIRST_POSITION: [],
        ONETEP_WRONG_FIRST_POSITION: [],
        ONETEP_CELL: [],
        ONETEP_ATOM_COUNT: [],
        ONETEP_IBFGS_IMPROVE: [],
        ONETEP_IBFGS_ITER: [],
        ONETEP_START_GEOM: [],
        ONETEP_RESUMING_GEOM: [],
        ONETEP_SPECIES: [],
        ONETEP_SPECIESL: [],
        ONETEP_FIRST_CELL: [],
        ONETEP_FIRST_CELLL: [],
        ONETEP_STRESS_CELL: []
    }

    # Index will be treated to get rid of duplicate or improving iterations
    output_corr = deepcopy(output)

    # Core properties that will be used in Yield
    properties = [ONETEP_TOTAL_ENERGY, ONETEP_FORCE, ONETEP_STRESS,
                  ONETEP_MULLIKEN, ONETEP_FIRST_CELL,
                  ONETEP_FIRST_CELLL]

    # Find all matches append them to the dictionary
    for idx, line in enumerate(fdo_lines):
        match = False
        for key in output:
            if key in line:
                match = key
                break
        if match:
            output[match].append(idx)
            # output[match].append(idx)
        # If a calculation died in the middle of nowhere...
        # Might be needed, keeping it here
        # if len(output[ONETEP_START]) - len(output[ONETEP_STOP]) > 1:
        #    output[ONETEP_STOP].append(i - 1)

    # Everything is numpy
    output = {key: np.array(value) for key, value in output.items()}
    # Conveniance notation (pointers: no overhead, no additional memory)
    ibfgs_iter = output[ONETEP_IBFGS_ITER]
    ibfgs_start = output[ONETEP_START_GEOM]
    ibfgs_improve = output[ONETEP_IBFGS_IMPROVE]
    ibfgs_resume = output[ONETEP_RESUMING_GEOM]
    i_first_positions = output[ONETEP_FIRST_POSITION]
    is_frac_positions = [i for i in i_first_positions if 'FRAC' in fdo_lines[i]]

    # In onetep species can have arbritary names,
    # We want to map them to real element names
    # Via the species block
    species = np.concatenate((output[ONETEP_SPECIES],
                              output[ONETEP_SPECIESL])).astype(np.int32)

    icells = np.hstack(
        (output[ONETEP_FIRST_CELL],
         output[ONETEP_FIRST_CELLL],
         output[ONETEP_STRESS_CELL])
    )
    icells = icells.astype(np.int32)
    # Using the fact that 0 == False and > 0 == True
    has_bfgs = len(ibfgs_iter)  \
        + len(output[ONETEP_START_GEOM]) \
        + len(output[ONETEP_RESUMING_GEOM])

    has_bfgs_improve = len(ibfgs_improve)
    has_bfgs_resume = len(ibfgs_resume)
    # When the input block position is written in lowercase
    # ONETEP does not print the initial position but a hash
    # of it, might be needed
    has_hash = len(output[ONETEP_WRONG_FIRST_POSITION])

    def is_in_bfgs(idx):
        """
        Check if a given index is in a BFGS block
        """
        for past, future in zip(output[ONETEP_START], np.hstack(
                (output[ONETEP_START][1:], [n_lines]))):
            if past < idx < future:
                if np.any((past < ibfgs_start) & (ibfgs_start < future)) or \
                        np.any((past < ibfgs_resume) & (ibfgs_resume < future)):
                    return True
        return False

    # If onetep has bfgs, the first atomic positions
    # Will be printed multiple times, we don't add them
    if has_bfgs:
        to_del = []
        for idx, tmp in enumerate(i_first_positions):
            if is_in_bfgs(tmp):
                to_del.append(idx)
        i_first_positions = np.delete(i_first_positions, to_del)

    ipositions = np.hstack((output[ONETEP_POSITION],
                            i_first_positions)).astype(np.int32)
    ipositions = np.sort(ipositions)

    n_pos = len(ipositions)

    # Some ONETEP files will not have any positions
    # due to how the software is coded. As a last
    # resort we look for a geom file with the same label.
    if n_pos == 0:
        name = fd.name
        label_maybe = basename(name).split('.')[0]
        geom_maybe = label_maybe + '.geom'
        if isfile(geom_maybe):
            from ase.io import read
            positions = read(geom_maybe, index="::",
                             format='castep-geom',
                             units={
                                 'Eh': units['Hartree'],
                                 'a0': units['Bohr']
                             }
                             )
            forces = [i.get_forces() for i in positions]
            has_bfgs = False
            has_bfgs_improve = False
            # way to make everything work
            ipositions = np.hstack(([0], output[ONETEP_IBFGS_ITER]))
        else:
            if has_hash:
                raise RuntimeError(no_positions_error)
            raise RuntimeError(unable_to_read)

    to_del = []

    # Important loop which:
    # - Get rid of improving BFGS iteration if improving == False
    # - Append None to properties to make sure each properties will
    # have the same length and each index correspond to the right
    # atomic configuration (hopefully).
    # Past is the index of the current atomic conf, future is the
    # index of the next one.
    for idx, (past, future) in enumerate(
            zip(ipositions, np.hstack((ipositions[1:], [n_lines])))):
        if has_bfgs:
            # BFGS resume prints the configuration at the beggining,
            # we don't want it
            if has_bfgs_resume:
                closest_resume = np.min(np.abs(past - ibfgs_resume))
                closest_starting = np.min(np.abs(past - ibfgs_iter))
                if closest_resume < closest_starting:
                    to_del.append(idx)
                    continue
            if has_bfgs_improve and not improving:
                # Find closest improve iteration index
                closest_improve = np.min(np.abs(past - ibfgs_improve))
                # Find closest normal iteration index
                closest_iter = np.min(np.abs(past - ibfgs_iter))
                if len(ibfgs_start):
                    closest_starting = np.min(np.abs(past - ibfgs_start))
                    closest = np.min([closest_iter, closest_starting])
                else:
                    closest = closest_iter
                # If improve is closer we delete
                if closest_improve < closest:
                    to_del.append(idx)
                    continue

        # We append None if no properties in contained for
        # one specific atomic configurations.
        for prop in properties:
            tmp, = np.where((past < output[prop]) & (output[prop] <= future))
            if len(tmp) == 0:
                output_corr[prop].append(None)
            else:
                output_corr[prop].extend(output[prop][tmp[:1]])

    # We effectively delete unwanted atomic configurations
    if to_del:
        new_indices = np.setdiff1d(np.arange(n_pos), to_del)
        ipositions = ipositions[new_indices]

    # Bunch of methods to grep properties from output.
    def parse_cell(idx):
        a, b, c = np.loadtxt([fdo_lines[idx + 2]]) * units['Bohr']
        al, be, ga = np.loadtxt([fdo_lines[idx + 4]])
        cell = Cell.fromcellpar([a, b, c, al, be, ga])
        return np.array(cell)

    def parse_charge(idx):
        n = 0
        offset = 4
        while idx + n < len(fdo_lines):
            if not fdo_lines[idx + n].strip():
                tmp_charges = np.loadtxt(
                    fdo_lines[idx + offset:idx + n - 1],
                    usecols=3)
                return tmp_charges
            n += 1
        return None
    # In ONETEP there is no way to differentiate electronic entropy
    # and entropy due to solvent, therefore there is no way to
    # extrapolate the energy at 0 K. We return the last energy
    # instead.
    def parse_energy(idx):
        n = 0
        energies = []
        while idx + n < len(fdo_lines):
            if '| Total' in fdo_lines[idx + n]:
                energies.append(float(fdo_lines[idx + n].split()[-2]))
            if 'LOCAL ENERGY COMPONENTS FROM MATRIX TRACES' in fdo_lines[idx + n]:
                return energies[-1] * units['Hartree']
            # Something is wrong with this ONETEP output
            if len(energies) > 2:
                raise RuntimeError('something is wrong with this ONETEP output')
            n += 1
        return None

    def parse_first_cell(idx):
        n = 0
        offset = 1
        while idx + n < len(fdo_lines):
            if 'ang' in fdo_lines[idx + n].lower():
                offset += 1
            if '%endblock lattice_cart' in fdo_lines[idx + n].lower():
                cell = np.loadtxt(
                    fdo_lines[idx + offset:idx + n]
                )
                return cell if offset == 2 else cell * units['Bohr']
            n += 1
        return None

    def parse_first_positions(idx):
        n = 0
        offset = 1
        while idx + n < len(fdo_lines):
            if 'ang' in fdo_lines[idx + n]:
                offset += 1
            if '%ENDBLOCK POSITIONS_' in fdo_lines[idx + n]:
                if 'FRAC' in fdo_lines[idx + n]:
                    conv_factor = 1
                else:
                    conv_factor = units['Bohr']
                tmp = np.loadtxt(fdo_lines[idx + offset:idx + n],
                                 dtype='str').reshape(-1, 4)
                els = np.char.array(tmp[:, 0])
                if offset == 2:
                    pos = tmp[:, 1:].astype(np.float32)
                else:
                    pos = tmp[:, 1:].astype(np.float32) * conv_factor
                try:
                    return Atoms(els, pos)
                # ASE doesn't recognize names used in ONETEP
                # as chemical symbol: dig deeper
                except KeyError:
                    tags, real_elements = find_correct_species(
                        els,
                        idx,
                        first=True
                    )
                    atoms = Atoms(real_elements, pos)
                    atoms.set_tags(tags)
                    return atoms
            n += 1
        return None

    def parse_force(idx):
        n = 0
        while idx + n < len(fdo_lines):
            if '* TOTAL:' in fdo_lines[idx + n]:
                tmp = np.loadtxt(fdo_lines[idx + 6:idx + n - 2],
                                 dtype=np.float64, usecols=(3, 4, 5))
                return tmp * units['Hartree'] / units['Bohr']
            n += 1
        return None

    def parse_positions(idx):
        n = 0
        offset = 7
        stop = 0
        while idx + n < len(fdo_lines):
            if 'xxxxxxxxxxxxxxxxxxxxxxxxx' in fdo_lines[idx + n].strip():
                stop += 1
            if stop == 2:
                tmp = np.loadtxt(fdo_lines[idx + offset:idx + n],
                                 dtype='str', usecols=(1, 3, 4, 5))
                els = np.char.array(tmp[:, 0])
                pos = tmp[:, 1:].astype(np.float32) * units['Bohr']
                try:
                    return Atoms(els, pos)
                # ASE doesn't recognize names used in ONETEP
                # as chemical symbol: dig deeper
                except KeyError:
                    tags, real_elements = find_correct_species(els, idx)
                    atoms = Atoms(real_elements, pos)
                    atoms.set_tags(tags)
                    return atoms
            n += 1
        return None

    def parse_species(idx):
        n = 1
        element_map = {}
        while idx + n < len(fdo_lines):
            sep = fdo_lines[idx + n].split()
            if '%endblock species ' in fdo_lines[idx + n].lower():
                return element_map
            element_map[sep[0]] = sep[1]
            n += 1
        return None

    def parse_spin(idx):
        n = 0
        offset = 4
        while idx + n < len(fdo_lines):
            if not fdo_lines[idx + n].strip():
                # If no spin is present we return None
                try:
                    tmp_spins = np.loadtxt(
                        fdo_lines[idx + offset:idx + n - 1],
                        usecols=4)
                except ValueError:
                    tmp_spins = None
                return tmp_spins
            n += 1
        return None

    # This is needed if ASE doesn't recognize the element
    def find_correct_species(els, idx, first=False):
        real_elements = []
        tags = []
        # Find nearest species block in case of
        # multi-output file with different species blocks.
        if first:
            closest_species = np.argmin(abs(idx - species))
        else:
            tmp = idx - species
            # Hopefully no one is ever gonna run a simulation
            # with more than 9999999999 species blocks
            tmp[tmp < 0] = 9999999999
            closest_species = np.argmin(tmp)
        elements_map = real_species[closest_species]
        for el in els:
            real_elements.append(elements_map[el])
            tag_maybe = el.replace(elements_map[el], '')
            if tag_maybe.isdigit():
                tags.append(int(tag_maybe))
            else:
                tags.append(False)
        return tags, real_elements

    cells = []
    for idx in icells:
        if idx in output[ONETEP_STRESS_CELL]:
            cell = parse_cell(idx) if idx else None
        else:
            cell = parse_first_cell(idx) if idx else None
        cells.append(cell)

    charges = []
    for idx in output_corr[ONETEP_MULLIKEN]:
        charge = parse_charge(idx) if idx else None
        charges.append(charge)

    energies = []
    for idx in output_corr[ONETEP_TOTAL_ENERGY]:
        energy = parse_energy(idx) if idx else None
        energies.append(energy)

    magmoms = []
    for idx in output_corr[ONETEP_MULLIKEN]:
        magmom = parse_spin(idx) if idx else None
        magmoms.append(magmom)

    real_species = []
    for idx in species:
        real_specie = parse_species(idx)
        real_species.append(real_specie)

    # If you are here and n_pos == 0 then it
    # means you read a CASTEP geom file (see line ~ 522)
    if n_pos > 0:
        positions, forces = [], []
        for idx in ipositions:
            if idx in i_first_positions:
                position = parse_first_positions(idx)
            else:
                position = parse_positions(idx)
            if position:
                positions.append(position)
            else:
                n_pos -= 1
                break
        for idx in output_corr[ONETEP_FORCE]:
            force = parse_force(idx) if idx else None
            forces.append(force)

    n_pos = len(positions)
    # Numpy trick to get rid of configuration that are essentially the same
    # in a regular geometry optimisation with internal BFGS, the first
    # configuration is printed three time, we get rid of it
    properties = [energies, forces, charges, magmoms]

    if has_bfgs:
        tmp = [i.positions for i in positions]
        to_del = []
        for i in range(len(tmp[:-1])):
            if is_in_bfgs(ipositions[i]):
                if np.array_equal(tmp[i], tmp[i + 1]):
                    # If the deleted configuration has a property
                    # we want to keep it
                    for prop in properties:
                        if np.any(prop[i + 1]) and not np.any(prop[i]):
                            prop[i] = prop[i + 1]
                    to_del.append(i + 1)
        c = np.full((len(tmp)), True)
        c[to_del] = False
        energies = [energies[i] for i in range(n_pos) if c[i]]
        forces = [forces[i] for i in range(n_pos) if c[i]]
        charges = [charges[i] for i in range(n_pos) if c[i]]
        positions = [positions[i] for i in range(n_pos) if c[i]]
        ipositions = [ipositions[i] for i in range(n_pos) if c[i]]
        magmoms = [magmoms[i] for i in range(n_pos) if c[i]]

    n_pos = len(positions)

    # Prepare atom objects with all the properties
    if isinstance(index, int):
        indices = [range(n_pos)[index]]
    else:
        indices = range(n_pos)[index]
    for idx in indices:
        if cells:
            tmp = ipositions[idx] - icells
            p, = np.where(tmp >= 0)
            tmp = tmp[p]
            closest_cell = np.argmin(tmp)
            cell = cells[p[closest_cell]]
            positions[idx].set_cell(cell)
            if ipositions[idx] in is_frac_positions:
                positions[idx].set_scaled_positions(
                    positions[idx].get_positions()
                )
        else:
            raise RuntimeError(
                'No cell found, are you sure this is a onetep output?')
        positions[idx].set_initial_charges(charges[idx])
        calc = SinglePointDFTCalculator(
            positions[idx],
            energy=energies[idx] if energies else None,
            free_energy=energies[idx] if energies else None,
            forces=forces[idx] if forces else None,
            charges=charges[idx] if charges else None,
            magmoms=magmoms[idx] if magmoms else None,
        )
        positions[idx].calc = calc
        # positions[idx].set_initial_charges(charges[idx])
        yield positions[idx]
