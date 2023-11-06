# from os.path import join
from unittest import mock
from unittest.mock import mock_open, patch, MagicMock

import pytest

from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.atoms import Atoms


@pytest.fixture
def vaspinput_factory():
    """Factory for GenerateVaspInput class, which mocks the generation of
    pseudopotentials."""

    def _vaspinput_factory(**kwargs) -> GenerateVaspInput:
        mocker = mock.Mock()
        inputs = GenerateVaspInput()
        inputs.set(**kwargs)
        inputs._build_pp_list = mocker(  # type: ignore[method-assign]
            return_value=None)
        return inputs

    return _vaspinput_factory


def get_mock_open_and_check_write(parameters, vaspinput_factory) -> MagicMock:
    mock = mock_open()
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.2]])
    calc_factory = vaspinput_factory(**parameters)
    calc_factory.initialize(atoms)
    with patch("ase.io.vasp_parsers.incar_writer.write_incar", mock):
        calc_factory.write_incar(atoms, "./")
        mock.assert_called_once_with(
            './',
            parameters,
            'INCAR created by Atomic Simulation Environment'
        )
    # with patch("ase.calculators.vasp.create_input.open", mock):
    #     calc_factory.write_incar(atoms, "./")
    #     mock.assert_called_once_with(join("./", 'INCAR'), "w")
        return mock


ASE_header = 'INCAR created by Atomic Simulation Environment\n'


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
    expected_output = ASE_header + " PREC = Low\n"
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)


def test_special_str_key(vaspinput_factory):
    parameters = {"xc": "PBE"}
    expected_output = ASE_header + " GGA = PE\n"
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)


def test_float_key(vaspinput_factory):
    parameters = {"encut": 400}
    expected_output = ASE_header + " ENCUT = 400.000000\n"
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)


def test_exp_key(vaspinput_factory):
    parameters = {"ediff": 1e-6}
    expected_output = ASE_header + " EDIFF = 1.00e-06\n"
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)


def test_int_key(vaspinput_factory):
    parameters = {"ibrion": 2}
    expected_output = ASE_header + " IBRION = 2\n"
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)


def test_list_bool_key(vaspinput_factory):
    parameters = {"lattice_constraints": [False, True, False]}
    expected_output = ASE_header + (" LATTICE_CONSTRAINTS = .FALSE. .TRUE. "
                                    ".FALSE.\n")
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)


def test_bool_key(vaspinput_factory):
    parameters = {"lhfcalc": True}
    expected_output = ASE_header + " LHFCALC = .TRUE.\n"
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)


def test_special_key(vaspinput_factory):
    parameters = {"lreal": True}
    expected_output = ASE_header + " LREAL = .TRUE.\n"
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)


def test_list_float_key(vaspinput_factory):
    parameters = {"magmom": [0.5, 1.5]}
    expected_output = ASE_header + (" MAGMOM = 1*0.5000 1*1.5000\n ISPIN = "
                                    "2\n")  # Writer uses :.4f
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)


def test_dict_key(vaspinput_factory):  # dict key. Current writer uses %.3f
    parameters = {"ldau_luj": {"H": {"L": 2, "U": 4.0, "J": 0.0}}}
    expected_output = ASE_header + (" LDAU = .TRUE.\n LDAUL = 2\n LDAUU = "
                                    "4.000\n LDAUJ = 0.000\n")
    check_last_call_to_write(parameters, expected_output, vaspinput_factory)
    # expected_output = [
    #     ASE_header,
    #     " LDAU = .TRUE.\n",
    #     " LDAUL = 2\n",
    #     " LDAUU = 4.000\n",
    #     " LDAUJ = 0.000\n",
    # ]
    # check_calls_to_write(parameters, expected_output, vaspinput_factory)
