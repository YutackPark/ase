import pytest
import numpy as np
from ase.geometry.bravais_type_engine import generate_niggli_op_table


ref_info = {
    'LINE': 1,
    'SQR': 1,
    'RECT': 1,
    'CRECT': 4,
    'HEX2D': 1,
    'OBL': 5,
    'FCC': 1,
    'BCC': 1,
    'CUB': 1,
    'TET': 2,
    'BCT': 5,
    'HEX': 2,
    'ORC': 1,
    'ORCC': 5,
    'ORCF': 2,
    'ORCI': 4,
    'RHL': 3,
    # 'MCL': 15,
    # 'MCLC': 27,
    # 'TRI': 19,
}


# We disable the three lattices that have infinite reductions.
# Maybe we can test those, but not today.
n3d_lattices = 14 - 3
n2d_lattices = 5
n1d_lattices = 1
assert len(ref_info) == n3d_lattices + n2d_lattices + n1d_lattices


def ref_info_iter():
    for key, val in ref_info.items():
        yield key, val


@pytest.mark.parametrize('lattice_name,ref_nops', ref_info_iter())
def test_generate_niggli_table(lattice_name, ref_nops):
    length_grid = np.logspace(-1, 1, 30)
    angle_grid = np.linspace(30, 170, 50)
    table = generate_niggli_op_table(lattices=[lattice_name],
                                     angle_grid=angle_grid,
                                     length_grid=length_grid)
    for key in table:
        print('{}: {}'.format(key, len(table[key])))

    mappings = table[lattice_name]
    assert len(mappings) == ref_nops
