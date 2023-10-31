from ase.io.vasp_parsers.kpoints_writer import write_kpoints
from unittest.mock import mock_open, patch
from collections import OrderedDict


def check_write_kpoints_file(parameters, expected_output):
    mock = mock_open()
    with patch("ase.io.vasp_parsers.kpoints_writer.open", mock):
        write_kpoints("directory", parameters)
        mock.assert_called_once_with("directory/KPOINTS", "w")
        kpoints = mock()
        kpoints.write.assert_called_once_with(expected_output)


def test_kpoints_Auto_mode():
    expected_output = """KPOINTS created by Atomic Simulation Environment
0
Auto
10"""
    parameters = {"Auto": 10}
    check_write_kpoints_file(parameters, expected_output)
    parameters = {"AUTO": 10}
    check_write_kpoints_file(parameters, expected_output)
    parameters = {"auto": 10}
    check_write_kpoints_file(parameters, expected_output)


def test_kpoints_Gamma_mode():
    expected_output = """KPOINTS created by Atomic Simulation Environment
0
Gamma
4 4 4"""
    parameters = {"Gamma": [4, 4, 4]}
    check_write_kpoints_file(parameters, expected_output)
    parameters = {"gamma": [4, 4, 4]}
    check_write_kpoints_file(parameters, expected_output)
    parameters = {"GAMMA": [4, 4, 4]}
    check_write_kpoints_file(parameters, expected_output)


def test_kpoints_Monkhorst_mode():
    expected_output = """KPOINTS created by Atomic Simulation Environment
0
Monkhorst
4 4 4"""
    parameters = {"Monkhorst": [4, 4, 4]}
    check_write_kpoints_file(parameters, expected_output)
    parameters = {"monkhorst": [4, 4, 4]}
    check_write_kpoints_file(parameters, expected_output)
    parameters = {"MONKHORST": [4, 4, 4]}
    check_write_kpoints_file(parameters, expected_output)


def test_kpoints_Line_reciprocal_mode():
    expected_output = """KPOINTS created by Atomic Simulation Environment
40
Line
Reciprocal
0 0 0
0.5 0.5 0
0.5 0.5 0
0.5 0.75 0.25
0.5 0.75 0.25
0 0 0"""
    parameters = {
        "Line": 40,
        "reciprocal": [
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.75, 0.25],
            [0.5, 0.75, 0.25],
            [0, 0, 0],
        ],
    }
    check_write_kpoints_file(parameters, expected_output)


def test_kpoints_Line_cartesian_mode():
    expected_output = """KPOINTS created by Atomic Simulation Environment
40
Line
Cartesian
0 0 0
0.5 0.5 0
0.5 0.5 0
0.5 0.75 0.25
0.5 0.75 0.25
0 0 0"""
    parameters = {
        "Line": 40,
        "CaRTESIAN": [
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.75, 0.25],
            [0.5, 0.75, 0.25],
            [0, 0, 0],
        ],
    }
    check_write_kpoints_file(parameters, expected_output)


def test_kpoints_Line_mode_incorrect_order():
    expected_output = """KPOINTS created by Atomic Simulation Environment
40
Line
Reciprocal
0 0 0
0.5 0.5 0
0.5 0.5 0
0.5 0.75 0.25
0.5 0.75 0.25
0 0 0"""
    parameters = {
        "reciprocal": [
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.75, 0.25],
            [0.5, 0.75, 0.25],
            [0, 0, 0],
        ],
        "Line": 40,
    }
    check_write_kpoints_file(parameters, expected_output)


def test_kpoints_explicit_reciprocal_mode():
    expected_output = """KPOINTS created by Atomic Simulation Environment
4
Reciprocal
0 0 0
0 0 0.5
0 0.5 0.5
0.5 0.5 0.5"""
    coordinates = [[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]]
    parameters = {"Reciprocal": coordinates}
    check_write_kpoints_file(parameters, expected_output)


def test_kpoints_explicit_cartesian_mode():
    expected_output = """KPOINTS created by Atomic Simulation Environment
4
Cartesian
0 0 0
0 0 0.5
0 0.5 0.5
0.5 0.5 0.5"""
    coordinates = [[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]]
    parameters = {"Cartesian": coordinates}
    check_write_kpoints_file(parameters, expected_output)


def test_kpoints_explicit_str_file():
    expected_output = """KPOINTS created by Atomic Simulation Environment
4
Cartesian
0 0 0
0 0 0.5
0 0.5 0.5
0.5 0.5 0.5"""
    check_write_kpoints_file(expected_output, expected_output)





