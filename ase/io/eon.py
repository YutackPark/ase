# Copyright (C) 2012-2023, Jesper Friis, SINTEF
# Copyright (C) 2024, Rohit Goswami, UI
# (see accompanying license files for ASE).
"""Module to read and write atoms EON reactant.con files.

See http://theory.cm.utexas.edu/eon/index.html for a description of EON.
"""
from warnings import warn
from pathlib import Path

import numpy as np

from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.utils import reader, writer

from dataclasses import dataclass
from typing import List, Tuple

ISOTOPE_TOL = 1e-8


@dataclass
class EONHeader:
    """
    A data class for storing header information of an EON file.

    Attributes
    ----------
    header_lines : List[str]
        The first two comment lines from the EON file header.
    cell_lengths : Tuple[float, float, float]
        The lengths of the cell vectors.
    cell_angles : Tuple[float, float, float]
        The angles between the cell vectors.
    Ncomponent : int
        The number of distinct atom types.
    component_counts : List[int]
        A list containing the count of atoms for each type.
    masses : List[float]
        A list containing the atomic masses for each type.
    """

    header_lines: List[str]
    # Actually these are float float float but.. mypy complains
    cell_lengths: Tuple[float, ...]
    cell_angles: Tuple[float, ...]
    Ncomponent: int
    component_counts: List[int]
    masses: List[float]


def process_header(lines: List[str]) -> EONHeader:
    """
    Processes the header lines from an EON file and returns an EONHeader object.

    This function parses the first 9 lines of an EON file, extracting
    information about the simulation cell, the number of atom types, their
    counts, and masses, and encapsulates this information in an EONHeader
    object.

    Parameters
    ----------
    lines : List[str]
        The first 9 lines of an EON file containing header information.

    Returns
    -------
    EONHeader
        An object containing the parsed header information.
    """
    header_lines = lines[:2]

    # Parse cell lengths and angles
    cell_lengths = tuple(map(float, lines[2].split()))
    cell_angles = tuple(map(float, lines[3].split()))
    if len(cell_lengths) != 3 or len(cell_angles) != 3:
        raise ValueError(
            "Cell lengths and angles must each contain exactly three values."
        )

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
    """
    Creates an Atoms object from coordinate blocks and header information.

    This function takes a list of coordinate blocks and the associated header
    information, constructs the cell, sets the atomic positions, masses, and
    optionally applies FixAtoms constraints based on the header information, and
    returns an ASE Atoms object.

    Parameters
    ----------
    coordblock : list of str
        The lines from an EON file representing atom coordinates and types.
    header : EONHeader
        The parsed header information.

    Returns
    -------
    Atoms
        An ASE Atoms object constructed from the given coordinate blocks and
        header.
    """
    symbols = []
    coords = []
    masses = []
    fixed = []
    # Ordering in EON is different from the ASE convention
    cell_angles = (
        header.cell_angles[2],
        header.cell_angles[1],
        header.cell_angles[0]
    )
    cellpar = [x for x in header.cell_lengths + cell_angles]
    for idx, nblock in enumerate(header.component_counts):
        elem_block = coordblock[:(nblock + 2)]
        symb = elem_block[0]
        symbols.extend(nblock * [symb])
        mass = header.masses[idx]
        masses.extend(nblock * [mass])
        for eline in elem_block[2:]:
            tokens = eline.split()
            coords.append([float(x) for x in tokens[:3]])
            fixed.append(bool(int(tokens[3])))
        coordblock = coordblock[(nblock + 2):]
    return Atoms(
        symbols=symbols,
        positions=coords,
        masses=masses,
        cell=cellpar_to_cell(cellpar),
        constraint=FixAtoms(mask=fixed),
    )


@reader
def read_eon(fileobj, index=-1):
    """
    Reads an EON file or directory and returns one or more Atoms objects.

    This function handles single EON files, in both single image and multi-image
    variants. It returns either a single Atoms object, a list of Atoms objects,
    or a specific Atoms object indexed from the file or directory.

    Parameters
    ----------
    fileobj : str or Path or file-like object
        The path to the EON file or directory, or an open file-like object.
    index : int, optional
        The index of the Atoms object to return. If -1 (default), returns all
        objects or a single object if only one is present.

    Returns
    -------
    Atoms or List[Atoms]
        Depending on the `index` parameter and the content of the fileobj,
        returns either a single Atoms object or a list of Atoms objects.
    """
    images = []
    while True:
        # Read and process headers if they exist
        try:
            lines = [next(fileobj).strip() for _ in range(9)]
        except StopIteration:
            break  # End of file
        header = process_header(lines)
        num_blocklines = (header.Ncomponent * 2) + sum(header.component_counts)
        coordblocks = [next(fileobj).strip() for _ in range(num_blocklines)]
        atoms = make_atoms(coordblocks, header)
        images.append(atoms)

    # XXX: I really don't like this since there should be a consistent return
    if index == -1:
        return images if len(images) > 1 else images[0]
    else:
        return images[index]


def read_states(states_dir: Path):
    """
    Reads all EON states from a directory and returns a list of Atoms objects.

    This function iterates through a directory expected to contain multiple
    states, each represented by its own directory named with digits. It reads
    the "reactant.con" file from each state directory and aggregates the
    resulting Atoms objects into a list.

    Parameters
    ----------
    states_dir : Path
        The directory containing state subdirectories.

    Returns
    -------
    List[Atoms]
        A list of Atoms objects read from each state directory.
    """
    subdirs = sorted(
        [x for x in states_dir.iterdir() if x.is_dir() and x.name.isdigit()],
        key=lambda d: int(d.name),
    )
    images = []
    for subdir in subdirs:
        reactant_path = subdir / "reactant.con"
        if reactant_path.exists():
            images.extend(read_eon(reactant_path, index=-1))
    return images


@writer
def write_eon(fileobj, images, comment="Generated by ASE"):
    """
    Writes structures to an EON trajectory file, allowing for multiple
    snapshots.

    This function iterates over all provided images, converting each one into a
    text format compatible with EON trajectory files. It handles the conversion
    of the cell parameters, chemical symbols, atomic masses, and atomic
    constraints. Only FixAtoms constraints are supported; any other constraints
    will generate a warning and be skipped. Arbitrary masses will raise an
    error, since EON will not accept them, i.e. no Duterium and Hydrogen.

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
    comment : str
        An additional comment statement which will be written out as the first
        header

    Raises
    ------
    Warning
        If any constraint in an Atoms object is not an instance of FixAtoms.
    RuntimeError
        If the masses for the species are not the default ones, i.e. if isotopes
        are present

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
            out.append(atoms.info.get("comment", comment))
        else:
            out.append(f"\n{atoms.info.get('comment', comment)}")
        out.append("preBox_header_2")  # ??

        a, b, c, alpha, beta, gamma = cell_to_cellpar(atoms.cell)
        out.append("%-10.6f  %-10.6f  %-10.6f" % (a, b, c))
        out.append("%-10.6f  %-10.6f  %-10.6f" % (gamma, beta, alpha))

        out.append("postBox_header_1")  # ??
        out.append("postBox_header_2")  # ??

        massdict = dict(zip(atoms.symbols, atoms.get_masses()))
        atomtypes = sorted(np.unique(atoms.symbols))
        atommasses = massdict.values()
        natoms = [atoms.symbols.count(at) for at in atomtypes]
        ntypes = len(atomtypes)

        for symbol in atoms.symbols:
            symbol_indices = atoms.symbols.indices()
            masses_for_symb = atoms.get_masses()[symbol_indices[symbol]]
            if masses_for_symb.ptp() > ISOTOPE_TOL:
                raise RuntimeError(
                    "Isotopes cannot be handled by EON"
                    ", ensure uniform masses for symbols"
                )

        out.append(str(ntypes))
        out.append(" ".join([str(n) for n in natoms]))
        out.append(" ".join([str(n) for n in atommasses]))

        atom_id = 0
        component_id = 0
        for species, indices in atoms.symbols.indices().items():
            fixed = np.array([False] * len(atoms))
            out.append(species)
            out.append("Coordinates of Component %d" % (component_id + 1))
            atom = atoms[indices]
            coords = atom.positions
            for constraint in atom.constraints:
                if not isinstance(constraint, FixAtoms):
                    warn(
                        "Only FixAtoms constraints are supported"
                        "by con files. Dropping %r",
                        constraint,
                    )
                    continue
                if constraint.index.dtype.kind == "b":
                    fixed = np.array(constraint.index, dtype=int)
                else:
                    fixed = np.zeros((natoms[component_id],), dtype=int)
                    for i in constraint.index:
                        fixed[i] = 1
            for xyz, fix in zip(coords, fixed):
                line_fmt = "{:>22.17f} {:>22.17f} {:>22.17f} {:d} {:4d}"
                line = line_fmt.format(*xyz, int(fix), atom_id)
                out.append(line)
                atom_id += 1
        fileobj.write("\n".join(out))
