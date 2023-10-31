from unittest import mock

import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.vasp.create_input import GenerateVaspInput, FLOAT_FORMAT
from ase.atoms import Atoms

calc = pytest.mark.calculator


@pytest.fixture
def rng():
    return np.random.RandomState(seed=42)


@pytest.fixture
def nacl(rng):
    atoms = bulk('NaCl', crystalstructure='rocksalt', a=4.1, cubic=True) * (
        3,
        3,
        3,
    )
    rng.shuffle(atoms.symbols)  # Ensure symbols are mixed
    return atoms


@pytest.fixture
def vaspinput_factory(nacl):
    """Factory for GenerateVaspInput class, which mocks the generation of
    pseudopotentials."""

    def _vaspinput_factory(atoms=None, **kwargs) -> GenerateVaspInput:
        if atoms is None:
            atoms = nacl
        mocker = mock.Mock()
        inputs = GenerateVaspInput()
        inputs.set(**kwargs)
        inputs._build_pp_list = mocker(return_value=None)  # type: ignore
        inputs.initialize(atoms)
        return inputs

    return _vaspinput_factory


@calc('vasp')
def test_vasp_incar(vaspinput_factory):
    """Test that INCAR is written correctly."""
    # Have each of every type of key in settings
    settings = {
        'xc': 'PBE',  # special str key
        'encut': 400,  # Float key. Current writer uses :5.6f
        'ediff': 1e-6,  # Exp key. Current writer uses :5.2e
        'ibrion': 2,  # Int key
        'prec': 'Low',  # str key
        'lattice_constraints': [False, True, False],  # list_bool key
        'iband': [1, 2, 3],  # list_int key
        'lhfcalc': True,  # bool key
        'lreal': True,  # special key
        'magmom': [0.5, 1.5],  # list_float key. Current writer uses :.4f
        'ldau_luj': {
            'H': {'L': 2, 'U': 4.0, 'J': 0.0},
        },  # dict key. Current writer uses %.3f
    }

    calc = vaspinput_factory(**settings)
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.2]])
    calc.initialize(atoms)
    calc.write_incar(atoms)

    # Check that INCAR is written correctly
    with open('INCAR', 'r') as f:
        lines = f.readlines()
    print(lines)
    assert (
        'INCAR created by Atomic Simulation Environment' == lines[0].strip()
    )

    assert f' ENCUT = {settings["encut"]:{FLOAT_FORMAT}}\n' in lines
