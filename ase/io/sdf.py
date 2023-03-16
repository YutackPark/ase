"""Reads chemical data in SDF format (wraps the molfile format).

See https://en.wikipedia.org/wiki/Chemical_table_file#SDF
"""
from typing import TextIO

from ase.atoms import Atoms
from ase.utils import reader


def get_num_atoms_sdf_v2000(first_line: str) -> int:
    """Parse the first line extracting the number of atoms."""
    return int(first_line[0:3])  # first three characters
    # http://biotech.fyicenter.com/1000024_SDF_File_Format_Specification.html


@reader
def read_sdf(file_obj: TextIO) -> Atoms:
    """Read the sdf data from a text file and compose the corresponding Atoms object."""
    lines = file_obj.readlines()
    # first three lines header
    del lines[:3]

    num_atoms = get_num_atoms_sdf_v2000(lines.pop(0))
    positions = []
    symbols = []
    for line in lines[:num_atoms]:
        ls = line.split()
        positions.append(tuple(map(float, ls[:3])))
        symbols.append(ls[3])

    return Atoms(symbols=symbols, positions=positions)
