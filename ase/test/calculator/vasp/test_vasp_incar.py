from os.path import join
from unittest import mock
from unittest.mock import mock_open, patch

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

def get_mock_open_and_check_write(parameters, vaspinput_factory):
    mock = mock_open()
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.2]])
    calc_factory = vaspinput_factory(**parameters)
    calc_factory.initialize(atoms)
    with patch("ase.calculators.vasp.create_input.open", mock):
        calc_factory.write_incar(atoms, "./")
        mock.assert_called_once_with(join("./", 'INCAR'), "w")
        return mock
        
def check_last_call_to_write(parameters, expected_output, vaspinput_factory):
    mock = get_mock_open_and_check_write(parameters, vaspinput_factory)
    incar = mock()
    incar.write.assert_called_with(expected_output)
    
def check_calls_to_write(parameters, expected_output_list, vaspinput_factory):
    mock = get_mock_open_and_check_write(parameters, vaspinput_factory)
    incar = mock()
    for output in expected_output_list:
        incar.write.assert_any_call(output)

def test_str_key(vaspinput_factory):
    parameters = {"prec": "Low"}
    expected_output = " PREC = Low\n"
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)

def test_special_str_key(vaspinput_factory):
    parameters = {"xc": "PBE"}
    expected_output = " GGA = PE\n"
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)
    
def test_float_key(vaspinput_factory):
    parameters = {"encut": 400}
    expected_output = " ENCUT = 400.000000\n"
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)
    
def test_exp_key(vaspinput_factory):
    parameters = {"ediff": 1e-6}
    expected_output = " EDIFF = 1.00e-06\n"
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)

def test_int_key(vaspinput_factory):
    parameters = {"ibrion": 2}
    expected_output = " IBRION = 2\n"
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)

def test_list_bool_key(vaspinput_factory):
    parameters = {"lattice_constraints": [False, True, False]}
    expected_output = " LATTICE_CONSTRAINTS = .FALSE. .TRUE. .FALSE.\n"
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)
    
def test_bool_key(vaspinput_factory):
    parameters = {"lhfcalc": True}
    expected_output = " LHFCALC = .TRUE.\n"
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)
    
def test_special_key(vaspinput_factory):    
    parameters = {"lreal": True}
    expected_output = " LREAL = .TRUE.\n"
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)
    
def test_list_float_key(vaspinput_factory):
    parameters = {"magmom": [0.5, 1.5]}
    expected_output = " MAGMOM = 1*0.5000 1*1.5000\n"  # list_float key. Current writer uses :.4f
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)

def test_dict_key(vaspinput_factory):  # dict key. Current writer uses %.3f
    parameters = {"ldau_luj": {"H": {"L": 2, "U": 4.0, "J": 0.0}}}
    # expected_output = " LDAUJ = 0.000\n"
    expected_output = [" LDAU = .TRUE.\n", " LDAUL = 2\n", " LDAUU = 4.000\n", " LDAUJ = 0.000\n"]
    check_calls_to_write(parameters, expected_output, vaspinput_factory)

    
