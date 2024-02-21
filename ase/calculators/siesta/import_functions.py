import numpy as np

from ase.io.fortranfile import FortranFile


def read_rho(fname):
    "Read unformatted Siesta charge density file"

    # TODO:
    #
    # Handle formatted and NetCDF files.
    #
    # Siesta source code (at least 2.0.2) can possibly also
    # save RHO as a _formatted_ file (the source code seems
    # prepared, but there seems to be no fdf-options for it though).
    # Siesta >= 3 has support for saving RHO as a NetCDF file
    # (according to manual)

    fh = FortranFile(fname)

    # Read (but ignore) unit cell vectors
    x = fh.readReals('d')
    if len(x) != 3 * 3:
        raise OSError('Failed to read cell vectors')

    # Read number of grid points and spin components
    x = fh.readInts()
    if len(x) != 4:
        raise OSError('Failed to read grid size')
    gpts = x  # number of 'X', 'Y', 'Z', 'spin' gridpoints

    rho = np.zeros(gpts)
    for ispin in range(gpts[3]):
        for n3 in range(gpts[2]):
            for n2 in range(gpts[1]):
                x = fh.readReals('f')
                if len(x) != gpts[0]:
                    raise OSError('Failed to read RHO[:,%i,%i,%i]' %
                                  (n2, n3, ispin))
                rho[:, n2, n3, ispin] = x

    fh.close()

    return rho


def readPLD(fname, norbitals, natoms):
    """
    Read unformatted siesta PLD file
    """
    import collections
    # use struct library to read mixed data type from binary
    import struct

    PLD_tuple = collections.namedtuple('PLD', ['max_rcut', 'orb2ao',
                                               'orb2uorb', 'orb2occ',
                                               'atm2sp', 'atm2shift',
                                               'coord_sc', 'cell',
                                               'nunit_cells'])

    fh = FortranFile(fname)

    orb2ao = np.zeros((norbitals), dtype=int)
    orb2uorb = np.zeros((norbitals), dtype=int)
    orb2occ = np.zeros((norbitals), dtype=float)

    max_rcut = fh.readReals('d')
    for iorb in range(norbitals):
        dat = fh.readRecord()
        dat_size = struct.calcsize('iid')
        val_list = struct.unpack('iid', dat[0:dat_size])
        orb2ao[iorb] = val_list[0]
        orb2uorb[iorb] = val_list[1]
        orb2occ[iorb] = val_list[2]

    atm2sp = np.zeros((natoms), dtype=int)
    atm2shift = np.zeros((natoms + 1), dtype=int)
    for iatm in range(natoms):
        atm2sp[iatm] = fh.readInts('i')[0]

    for iatm in range(natoms + 1):
        atm2shift[iatm] = fh.readInts('i')[0]

    cell = np.zeros((3, 3), dtype=float)
    nunit_cells = np.zeros((3), dtype=int)
    for i in range(3):
        cell[i, :] = fh.readReals('d')
    nunit_cells = fh.readInts('i')

    coord_sc = np.zeros((natoms, 3), dtype=float)
    for iatm in range(natoms):
        coord_sc[iatm, :] = fh.readReals('d')

    fh.close()
    return PLD_tuple(max_rcut, orb2ao, orb2uorb, orb2occ, atm2sp, atm2shift,
                     coord_sc, cell, nunit_cells)
