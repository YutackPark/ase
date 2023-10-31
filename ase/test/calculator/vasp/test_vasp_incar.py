from os.path import join
from unittest import mock

import pytest

from ase.calculators.vasp.create_input import GenerateVaspInput, FLOAT_FORMAT, EXP_FORMAT
from ase.atoms import Atoms


@pytest.fixture
def vaspinput_factory():
    """Factory for GenerateVaspInput class, which mocks the generation of
    pseudopotentials."""

    def _vaspinput_factory(**kwargs) -> GenerateVaspInput:
        mocker = mock.Mock()
        inputs = GenerateVaspInput()
        inputs.set(**kwargs)
        inputs._build_pp_list = mocker(return_value=None)  # type: ignore
        return inputs

    return _vaspinput_factory


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

    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.2]])
    calc_factory = vaspinput_factory(**settings)
    calc_factory.initialize(atoms)
    calc_factory.write_incar(atoms)

    # Check that INCAR is written correctly
    with open('INCAR', 'r') as f:
        lines = f.readlines()
    print(lines)
    assert (
        'INCAR created by Atomic Simulation Environment' == lines[0].strip()
    )

    assert f' ENCUT = {settings["encut"]:{FLOAT_FORMAT}}\n' in lines
    assert f' EDIFF = {settings["ediff"]:{EXP_FORMAT}}\n' in lines
    assert f' IBRION = {settings["ibrion"]}\n' in lines
    assert f' PREC = {settings["prec"]}\n' in lines
    # assert f' LHFCALC = {settings["lhfcalc"]}\n' in lines
    # assert f' LREAL = {settings["lreal"]}\n' in lines
    # assert f' MAGMOM = {settings["magmom"][0]:.4f} {settings["magmom"][1]:.4f}\n' in lines

    
from unittest.mock import mock_open, patch

def test_open_incar(vaspinput_factory):
    mock = mock_open()
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

    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.2]])
    calc_factory = vaspinput_factory(**settings)
    calc_factory.initialize(atoms)
    with patch("ase.calculators.vasp.create_input", mock):
        calc_factory.write_incar(atoms)
        mock.assert_called_once_with(join(directory, 'INCAR'), "w")
        

