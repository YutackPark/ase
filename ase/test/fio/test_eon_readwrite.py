"""Check that reading and writing .con files is consistent."""

import numpy as np
import numpy.testing as npt

import ase
import ase.io
import ase.symbols

# Error tolerance.
TOL = 1e-6

# The corresponding data as an ASE Atoms object.
DATA = ase.Atoms(
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
    npt.assert_allclose(box.cell, DATA.cell, rtol=TOL, atol=0)
    assert (box.symbols == ase.symbols.string2symbols("Cu3")).all() == True
    npt.assert_allclose(box.get_masses(), np.array([63.5459999] * 3), rtol=TOL)
    npt.assert_allclose(box.positions, DATA.positions, rtol=TOL)


def test_eon_write_single(datadir):
    out_file = "out.con"
    ase.io.write(out_file, DATA, format="eon")
    data2 = ase.io.read(out_file, format="eon")
    npt.assert_allclose(data2.cell, DATA.cell, rtol=TOL, atol=0)
    npt.assert_allclose(data2.positions, DATA.positions, rtol=TOL)


def test_eon_read_multi(datadir):
    images = ase.io.read(f"{datadir}/io/con/multi.con", format="eon", index=":")
    assert len(images) == 10
    npt.assert_allclose(
        images[0].constraints[0].get_indices(), np.array([0, 1]), rtol=1e-5, atol=0
    )
    npt.assert_allclose(
        images[1].constraints[0].get_indices(), np.array([]), rtol=1e-5, atol=0
    )
