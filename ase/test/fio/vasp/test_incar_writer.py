from ase.io.vasp_parsers.incar_writer import write_incar
from unittest.mock import mock_open, patch


def test_write_string_to_incar():
    parameters = {"INCAR_TAG": "string"}
    expected_output = "INCAR_TAG = string"
    check_write_incar_file(parameters, expected_output)


def check_write_incar_file(parameters, expected_output):
    mock = mock_open()
    with patch("ase.io.vasp_parsers.incar_writer.open", mock):
        write_incar(parameters)
        mock.assert_called_once_with("INCAR", "w")
        incar = mock()
        incar.write.assert_called_once_with(expected_output)


def test_write_integer_to_incar():
    parameters = {"INCAR_TAG": 5}
    expected_output = "INCAR_TAG = 5"
    check_write_incar_file(parameters, expected_output)


def test_write_bool_to_incar():
    parameters = {"INCAR_TAG": True}
    expected_output = "INCAR_TAG = True"
    check_write_incar_file(parameters, expected_output)
    parameters = {"INCAR_TAG": False}
    expected_output = "INCAR_TAG = False"
    check_write_incar_file(parameters, expected_output)
