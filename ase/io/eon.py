# Copyright (C) 2012, Jesper Friis, SINTEF
# (see accompanying license files for ASE).
"""Module to read and write atoms EON reactant.con files.

See http://theory.cm.utexas.edu/eon/index.html for a description of EON.
"""
import os
from glob import glob
from warnings import warn
from pathlib import Path

import numpy as np

from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.utils import writer

from dataclasses import dataclass, field
from typing import List, Tuple


def print_con_atom_header(ostring, ntypes, natoms, atommasses):
    ostring.append(str(ntypes))
    ostring.append(" ".join([str(n) for n in natoms]))
    ostring.append(" ".join([str(n) for n in atommasses]))


@dataclass
class EONHeader:
    header_lines: List[str]
    cell_lengths: Tuple[float, float, float]
    cell_angles: Tuple[float, float, float]
    Ncomponent: int
    component_counts: List[int]
    masses: List[float]


def process_header(lines: List[str]) -> EONHeader:
    header_lines = lines[:2]  # Assuming first two lines are header descriptions

    # Parse cell lengths and angles
    cell_lengths = tuple(map(float, lines[2].split()))
    cell_angles = tuple(map(float, lines[3].split()))

    # Parse number of components
    Ncomponent = int(lines[6])
    component_counts = list(map(int, lines[7].split()))
    masses = list(map(float, lines[8].split()))

    return EONHeader(
        header_lines=header_lines,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
        Ncomponent=Ncomponent,
        component_counts=component_counts,
        masses=masses,
    )


def make_atoms(coordblock, header):
    symbols = []
    coords = []
    masses = []
    fixed = []
    cellpar = [float(x) for x in header.cell_lengths + header.cell_angles]
    for idx, nblock in enumerate(header.component_counts):
        elem_block = coordblock[: nblock + 2]
        symb = elem_block[0]
        symbols.extend(nblock * [symb])
        mass = header.masses[idx]
        masses.extend(nblock * [mass])
        for eline in elem_block[2:]:
            tokens = eline.split()
            coords.append([float(x) for x in tokens[:3]])
            fixed.append(bool(int(tokens[3])))
        coordblock = coordblock[nblock + 2 :]
    return Atoms(
        symbols=symbols,
        positions=coords,
        masses=masses,
        cell=cellpar_to_cell(cellpar),
        constraint=FixAtoms(mask=fixed),
        info=dict(comment="\n".join(header.header_lines)),
    )


def read_eon2(file_path, index=-1):
    images = []
    with open(file_path, "r") as fd:
        while True:
            # Read and process headers if they exist
            try:
                lines = [next(fd).strip() for _ in range(9)]  # Header is 9 lines
            except StopIteration:
                break  # End of file
            header = process_header(lines)
            num_blocklines = (header.Ncomponent * 2) + sum(header.component_counts)
            coordblocks = [next(fd).strip() for _ in range(num_blocklines)]
            atoms = make_atoms(coordblocks, header)
            images.append(atoms)

    # XXX: I really don't like this..
    if index == -1:
        if len(images) == 1:
            return images[0]
        else:
            return images
    else:
        return images[index]


def read_eon(fileobj, index=-1):
    """Reads an EON reactant.con file.  If *fileobj* is the name of a
    "states" directory created by EON, all the structures will be read."""
    if isinstance(fileobj, str):
        if os.path.isdir(fileobj):
            return read_states(fileobj)
        else:
            fd = open(fileobj)
    else:
        fd = fileobj

    more_images_to_read = True
    images = []

    first_line = fd.readline()
    while more_images_to_read:

        comment = first_line.strip()
        fd.readline()  # 0.0000 TIME  (??)
        cell_lengths = fd.readline().split()
        cell_angles = fd.readline().split()
        # Different order of angles in EON.
        cell_angles = [cell_angles[2], cell_angles[1], cell_angles[0]]
        cellpar = [float(x) for x in cell_lengths + cell_angles]
        fd.readline()  # 0 0     (??)
        fd.readline()  # 0 0 0   (??)
        ntypes = int(fd.readline())  # number of atom types
        natoms = [int(n) for n in fd.readline().split()]
        atommasses = [float(m) for m in fd.readline().split()]

        symbols = []
        coords = []
        masses = []
        fixed = []
        for n in range(ntypes):
            symbol = fd.readline().strip()
            symbols.extend([symbol] * natoms[n])
            masses.extend([atommasses[n]] * natoms[n])
            fd.readline()  # Coordinates of Component n
            for i in range(natoms[n]):
                row = fd.readline().split()
                coords.append([float(x) for x in row[:3]])
                fixed.append(bool(int(row[3])))

        atoms = Atoms(symbols=symbols,
                      positions=coords,
                      masses=masses,
                      cell=cellpar_to_cell(cellpar),
                      constraint=FixAtoms(mask=fixed),
                      info=dict(comment=comment))

        images.append(atoms)

        first_line = fd.readline()
        if first_line == '':
            more_images_to_read = False

    if isinstance(fileobj, str):
        fd.close()

    if not index:
        return images
    else:
        return images[index]


def read_states(states_dir):
    """Read structures stored by EON in the states directory *states_dir*."""
    subdirs = glob(os.path.join(states_dir, '[0123456789]*'))
    subdirs.sort(key=lambda d: int(os.path.basename(d)))
    images = [read_eon(os.path.join(subdir, 'reactant.con'))
              for subdir in subdirs]
    return images


@writer
def write_eon(fileobj, images):
    """Writes structure to EON reactant.con file
    Multiple snapshots are not allowed."""
    if isinstance(images, Atoms):
        atoms = images
    elif len(images) == 1:
        atoms = images[0]
    else:
        raise ValueError('Can only write one configuration to EON '
                         'reactant.con file')

    out = []
    out.append(atoms.info.get('comment', 'Generated by ASE'))
    out.append('0.0000 TIME')  # ??

    a, b, c, alpha, beta, gamma = cell_to_cellpar(atoms.cell)
    out.append(f'{a:<10.6f}  {b:<10.6f}  {c:<10.6f}')
    out.append(f'{gamma:<10.6f}  {beta:<10.6f}  {alpha:<10.6f}')

    out.append('0 0')    # ??
    out.append('0 0 0')  # ??

    symbols = atoms.get_chemical_symbols()
    massdict = dict(list(zip(symbols, atoms.get_masses())))
    atomtypes = sorted(massdict.keys())
    atommasses = [massdict[at] for at in atomtypes]
    natoms = [symbols.count(at) for at in atomtypes]
    ntypes = len(atomtypes)

    print_con_atom_header(out, ntypes, natoms, atommasses)

    atom_id = 0
    for n in range(ntypes):
        fixed = np.array([False] * len(atoms))
        out.append(atomtypes[n])
        out.append('Coordinates of Component %d' % (n + 1))
        indices = [i for i, at in enumerate(symbols) if at == atomtypes[n]]
        a = atoms[indices]
        coords = a.positions
        for c in a.constraints:
            if not isinstance(c, FixAtoms):
                warn('Only FixAtoms constraints are supported by con files. '
                     'Dropping %r', c)
                continue
            if c.index.dtype.kind == 'b':
                fixed = np.array(c.index, dtype=int)
            else:
                fixed = np.zeros((natoms[n], ), dtype=int)
                for i in c.index:
                    fixed[i] = 1
        for xyz, fix in zip(coords, fixed):
            out.append('%22.17f %22.17f %22.17f %d %4d' %
                       (tuple(xyz) + (fix, atom_id)))
            atom_id += 1
    fileobj.write('\n'.join(out))
    fileobj.write('\n')


@writer
def write_eon_traj(fileobj, images):
    """
    Writes structures to an EON trajectory file, allowing for multiple
    snapshots.

    This function iterates over all provided images, converting each one into a
    text format compatible with EON trajectory files. It handles the conversion
    of the cell parameters, chemical symbols, atomic masses, and atomic
    constraints. Only FixAtoms constraints are supported; any other constraints
    will generate a warning and be skipped.

    Parameters
    ----------
    fileobj : file object
        An opened, writable file or file-like object to which the EON trajectory
        information will be written.
    images : list of Atoms
        A list of ASE Atoms objects representing the images (atomic structures)
        to be written into the EON trajectory file. Each Atoms object should
        have a cell attribute and may have a constraints attribute. If the
        constraints attribute is not a FixAtoms object, a warning will be
        issued.

    Raises
    ------
    Warning
        If any constraint in an Atoms object is not an instance of FixAtoms.

    Returns
    -------
    None
        The function writes directly to the provided file object.

    See Also
    --------
    ase.io.utils.segment_list : for segmenting the list of images.

    Examples
    --------
    >>> from ase.io import Trajectory
    >>> from ase.io.utils import segment_list
    >>> traj = Trajectory("neb.traj")
    >>> n_images = 10  # Segment size, i.e. number of images in the NEB
    >>> for idx, pth in enumerate(segment_list(traj, n_images)):
    ...     with open(f"outputs/neb_path_{idx:03d}.con", "w") as fobj:
    ...         write_eon_traj(fobj, pth)
    """

    for idx, atoms in enumerate(images):
        out = []
        if idx == 0:
            out.append(atoms.info.get('comment', 'Generated by ASE'))
        else:
            out.append(f"\n{atoms.info.get('comment', 'Generated by ASE')}")
        out.append('preBox_header_2')  # ??

        a, b, c, alpha, beta, gamma = cell_to_cellpar(atoms.cell)
        out.append('%-10.6f  %-10.6f  %-10.6f' % (a, b, c))
        out.append('%-10.6f  %-10.6f  %-10.6f' % (gamma, beta, alpha))

        out.append('postBox_header_1')    # ??
        out.append('postBox_header_2')  # ??

        symbols = atoms.get_chemical_symbols()
        massdict = dict(list(zip(symbols, atoms.get_masses())))
        atomtypes = sorted(massdict.keys())
        atommasses = [massdict[at] for at in atomtypes]
        natoms = [symbols.count(at) for at in atomtypes]
        ntypes = len(atomtypes)

        print_con_atom_header(out, ntypes, natoms, atommasses)

        atom_id = 0
        for n in range(ntypes):
            fixed = np.array([False] * len(atoms))
            out.append(atomtypes[n])
            out.append('Coordinates of Component %d' % (n + 1))
            indices = [i for i, at in enumerate(symbols) if at == atomtypes[n]]
            a = atoms[indices]
            coords = a.positions
            for c in a.constraints:
                if not isinstance(c, FixAtoms):
                    warn('Only FixAtoms constraints are supported'
                         'by con files. Dropping %r', c)
                    continue
                if c.index.dtype.kind == 'b':
                    fixed = np.array(c.index, dtype=int)
                else:
                    fixed = np.zeros((natoms[n], ), dtype=int)
                    for i in c.index:
                        fixed[i] = 1
            for xyz, fix in zip(coords, fixed):
                out.append('%22.17f %22.17f %22.17f %d %4d' %
                           (tuple(xyz) + (fix, atom_id)))
                atom_id += 1
        fileobj.write('\n'.join(out))
