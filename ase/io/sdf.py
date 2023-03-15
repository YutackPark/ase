"""Reads chemical data in SDF format (wraps the molfile format).

See https://en.wikipedia.org/wiki/Chemical_table_file#SDF
"""
from typing import TextIO

from ase.atoms import Atoms
from ase.utils import reader


@reader
def read_sdf(file_obj: TextIO) -> Atoms:
    lines = file_obj.readlines()
    # first three lines header
    del lines[:3]

    first_line = lines.pop(0)
    num_atoms = int(first_line[0:3])  # first three characters
    # http://biotech.fyicenter.com/1000024_SDF_File_Format_Specification.html
    positions = []
    symbols = []
    for line in lines[:num_atoms]:
        x, y, z, symbol = line.split()[:4]
        symbols.append(symbol)
        positions.append([float(x), float(y), float(z)])
    return Atoms(symbols=symbols, positions=positions)
