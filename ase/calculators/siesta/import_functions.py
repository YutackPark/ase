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


def readDIM(fname):
    """
    Read unformatted siesta DIM file
    """
    import collections

    DIM_tuple = collections.namedtuple('DIM', ['natoms_sc', 'norbitals_sc',
                                               'norbitals', 'nspin',
                                               'nnonzero',
                                               'natoms_interacting'])

    fh = FortranFile(fname)

    natoms_sc = fh.readInts('i')[0]
    norbitals_sc = fh.readInts('i')[0]
    norbitals = fh.readInts('i')[0]
    nspin = fh.readInts('i')[0]
    nnonzero = fh.readInts('i')[0]
    natoms_interacting = fh.readInts('i')[0]
    fh.close()

    return DIM_tuple(natoms_sc, norbitals_sc, norbitals, nspin,
                     nnonzero, natoms_interacting)


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


def readWFSX(fname):
    """
    Read unformatted siesta WFSX file
    """
    import collections
    # use struct library to read mixed data type from binary
    import struct

    WFSX_tuple = collections.namedtuple('WFSX',
                                        ['nkpoints', 'nspin', 'norbitals',
                                         'gamma', 'orb2atm', 'orb2strspecies',
                                         'orb2ao', 'orb2n', 'orb2strsym',
                                         'kpoints', 'DFT_E', 'DFT_X',
                                         'mo_spin_kpoint_2_is_read'])

    fh = FortranFile(fname)

    nkpoints, gamma = fh.readInts('i')
    nspin = fh.readInts('i')[0]
    norbitals = fh.readInts('i')[0]

    orb2atm = np.zeros((norbitals), dtype=int)
    orb2strspecies = []
    orb2ao = np.zeros((norbitals), dtype=int)
    orb2n = np.zeros((norbitals), dtype=int)
    orb2strsym = []
    # for string list are better to select all the string length

    dat_size = struct.calcsize('i20sii20s')
    dat = fh.readRecord()

    ind_st = 0
    ind_fn = dat_size
    for iorb in range(norbitals):
        val_list = struct.unpack('i20sii20s', dat[ind_st:ind_fn])
        orb2atm[iorb] = val_list[0]
        orb2strspecies.append(val_list[1])
        orb2ao[iorb] = val_list[2]
        orb2n[iorb] = val_list[3]
        orb2strsym.append(val_list[4])
        ind_st = ind_st + dat_size
        ind_fn = ind_fn + dat_size
    orb2strspecies = np.array(orb2strspecies)
    orb2strsym = np.array(orb2strsym)

    kpoints = np.zeros((3, nkpoints), dtype=np.float64)
    DFT_E = np.zeros((norbitals, nspin, nkpoints), dtype=np.float64)

    if (gamma == 1):
        DFT_X = np.zeros((1, norbitals, norbitals, nspin, nkpoints),
                         dtype=np.float64)
        eigenvector = np.zeros((1, norbitals), dtype=float)
    else:
        DFT_X = np.zeros((2, norbitals, norbitals, nspin, nkpoints),
                         dtype=np.float64)
        eigenvector = np.zeros((2, norbitals), dtype=float)

    mo_spin_kpoint_2_is_read = np.zeros((norbitals, nspin, nkpoints),
                                        dtype=bool)
    mo_spin_kpoint_2_is_read[0:norbitals, 0:nspin, 0:nkpoints] = False

    dat_size = struct.calcsize('iddd')
    for ikpoint in range(nkpoints):
        for ispin in range(nspin):
            dat = fh.readRecord()
            val_list = struct.unpack('iddd', dat[0:dat_size])
            ikpoint_in = val_list[0] - 1
            kpoints[0:3, ikpoint] = val_list[1:4]
            if (ikpoint != ikpoint_in):
                raise ValueError('siesta_get_wfsx: ikpoint != ikpoint_in')
            ispin_in = fh.readInts('i')[0] - 1
            if (ispin_in > nspin - 1):
                msg = 'siesta_get_wfsx: err: ispin_in>nspin\n \
                     siesta_get_wfsx: ikpoint, ispin, ispin_in = \
                     {}  {}  {}\n siesta_get_wfsx'.format(ikpoint,
                                                          ispin, ispin_in)
                raise ValueError(msg)

            norbitals_in = fh.readInts('i')[0]
            if (norbitals_in > norbitals):
                msg = 'siesta_get_wfsx: err: norbitals_in>norbitals\n \
                     siesta_get_wfsx: ikpoint, norbitals, norbitals_in = \
                     {}  {}  {}\n siesta_get_wfsx'.format(ikpoint,
                                                          norbitals,
                                                          norbitals_in)
                raise ValueError(msg)

            for imolecular_orb in range(norbitals_in):
                imolecular_orb_in = fh.readInts('i')[0] - 1
                if (imolecular_orb_in > norbitals - 1):
                    msg = """
                        siesta_get_wfsx: err: imolecular_orb_in>norbitals\n
                        siesta_get_wfsx: ikpoint, norbitals,
                        imolecular_orb_in = {}  {}  {}\n
                        siesta_get_wfsx""".format(ikpoint, norbitals,
                                                  imolecular_orb_in)
                    raise ValueError(msg)

                real_E_eV = fh.readReals('d')[0]
                eigenvector = fh.readReals('f')
                DFT_E[imolecular_orb_in, ispin_in,
                      ikpoint] = real_E_eV / 13.60580
                DFT_X[:, :, imolecular_orb_in, ispin_in,
                      ikpoint] = eigenvector
                mo_spin_kpoint_2_is_read[imolecular_orb_in, ispin_in,
                                         ikpoint] = True

            if (not all(mo_spin_kpoint_2_is_read[:, ispin_in, ikpoint])):
                msg = 'siesta_get_wfsx: warn: .not. all(mo_spin_k_2_is_read)'
                print('mo_spin_kpoint_2_is_read = ', mo_spin_kpoint_2_is_read)
                raise ValueError(msg)

    fh.close()
    return WFSX_tuple(nkpoints, nspin, norbitals, gamma, orb2atm,
                      orb2strspecies, orb2ao, orb2n, orb2strsym,
                      kpoints, DFT_E, DFT_X, mo_spin_kpoint_2_is_read)
