import os
import re
from io import StringIO
from pathlib import Path

import numpy as np

from ase.io import read
from ase.units import Bohr, Hartree
from ase.utils import reader, writer

# Made from NWChem interface


@reader
def read_geom_orcainp(fd):
    """Method to read geometry from an ORCA input file."""
    lines = fd.readlines()

    # Find geometry region of input file.
    stopline = 0
    for index, line in enumerate(lines):
        if line[1:].startswith('xyz '):
            startline = index + 1
            stopline = -1
        elif (line.startswith('end') and stopline == -1):
            stopline = index
        elif (line.startswith('*') and stopline == -1):
            stopline = index
    # Format and send to read_xyz.
    xyz_text = '%i\n' % (stopline - startline)
    xyz_text += ' geometry\n'
    for line in lines[startline:stopline]:
        xyz_text += line
    atoms = read(StringIO(xyz_text), format='xyz')
    atoms.set_cell((0., 0., 0.))  # no unit cell defined

    return atoms


@writer
def write_orca(fd, atoms, params):
    # conventional filename: '<name>.inp'
    fd.write(f"! {params['orcasimpleinput']} \n")
    fd.write(f"{params['orcablocks']} \n")

    if 'coords' not in params['orcablocks']:
        fd.write('*xyz')
        fd.write(" %d" % params['charge'])
        fd.write(" %d \n" % params['mult'])
        for atom in atoms:
            if atom.tag == 71:  # 71 is ascii G (Ghost)
                symbol = atom.symbol + ' : '
            else:
                symbol = atom.symbol + '   '
            fd.write(
                symbol
                + str(atom.position[0])
                + " "
                + str(atom.position[1])
                + " "
                + str(atom.position[2])
                + "\n"
            )
        fd.write('*\n')


@reader
def read_orca_energy(fd):
    """Read Energy from ORCA output file."""
    text = fd.read()
    re_energy = re.compile(r"FINAL SINGLE POINT ENERGY.*\n")
    re_not_converged = re.compile(r"Wavefunction not fully converged")

    found_line = re_energy.finditer(text)
    energy = float('nan')
    for match in found_line:
        if not re_not_converged.search(match.group()):
            energy = float(match.group().split()[-1]) * Hartree
    if np.isnan(energy):
        raise RuntimeError('No energy')
    else:
        return energy


@reader
def read_orca_forces(fd):
    """Read Forces from ORCA output file."""
    getgrad = False
    gradients = []
    tempgrad = []
    for i, line in enumerate(fd):
        if line.find('# The current gradient') >= 0:
            getgrad = True
            gradients = []
            tempgrad = []
            continue
        if getgrad and "#" not in line:
            grad = line.split()[-1]
            tempgrad.append(float(grad))
            if len(tempgrad) == 3:
                gradients.append(tempgrad)
                tempgrad = []
        if '# The at' in line:
            getgrad = False

    forces = -np.array(gradients) * Hartree / Bohr
    return forces


def read_orca_outputs(directory, stdout_path):
    results = {}
    energy = read_orca_energy(Path(stdout_path))
    results['energy'] = energy
    results['free_energy'] = energy

    # Does engrad always exist? - No!
    # Will there be other files -No -> We should just take engrad
    # as a direct argument.  Or maybe this function does not even need to
    # exist.
    engrad_path = Path(stdout_path).with_suffix('.engrad')
    if os.path.isfile(engrad_path):
        results['forces'] = read_orca_forces(engrad_path)
    return results
