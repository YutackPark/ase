"""Check that reading and writing .con files is consistent."""

import numpy as np

import ase
import ase.io
import ase.symbols

# Error tolerance.
TOL = 1e-6

# The corresponding data as an ASE Atoms object.
data = ase.Atoms(
    "Cu3",
    cell=np.array([[7.22, 0, 0], [1, 10.83, 0], [1, 1, 14.44]]),
    positions=np.array(
        [
            [1.04833333, 0.965, 0.9025],
            [3.02, 2.77, 0.9025],
            [6.36666667, 10.865, 13.5375],
        ]
    ),
    pbc=(True, True, True),
)


def test_eon_read_single(datadir):
    box = ase.io.read(f"{datadir}/io/con/single.con", format="eon")
    # Check cell vectors.
    assert (abs(box.cell - data.cell)).sum() < TOL  # read: cell vector check
    # Check atom positions.
    # read: position check
    assert (abs(box.positions - data.positions)).sum() < TOL
    assert (box.symbols == ase.symbols.string2symbols("Cu3")).all() == True
    assert abs(box.get_masses() - np.array([63.5459999] * 3)).sum() < TOL

def test_eon_write_single(datadir):
    out_file = "out.con"
    ase.io.write(out_file, data, format="eon")
    data2 = ase.io.read(out_file, format="eon")
    # Check cell vectors.
    # write: cell vector check
    assert (abs(data2.cell - data.cell)).sum() < TOL
    # Check atom positions.
    # write: position check
    assert (abs(data2.positions - data.positions)).sum() < TOL
