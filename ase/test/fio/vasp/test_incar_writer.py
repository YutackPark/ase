from ase.io.vasp_parsers.incar_writer import write_incar
from unittest.mock import mock_open, patch


def test_write_string_to_incar():
    mock = mock_open()
    with patch("ase.io.vasp_parsers.incar_writer.open", mock):
        write_incar({"INCAR_TAG": "string"})
        mock.assert_called_once_with("INCAR", "w")
        incar = mock()
        incar.write.assert_called_once_with("INCAR_TAG = string")


def test_write_integer_to_incar():
    mock = mock_open()
    with patch("ase.io.vasp_parsers.incar_writer.open", mock):
        write_incar({"INCAR_TAG": 5})
        mock.assert_called_once_with("INCAR", "w")
        incar = mock()
        incar.write.assert_called_once_with("INCAR_TAG = 5") 
                                                                                                   

def test_write_True_to_incar():
    mock = mock_open()
    with patch("ase.io.vasp_parsers.incar_writer.open", mock):
        write_incar({"INCAR_TAG": True})
        mock.assert_called_once_with("INCAR", "w")
        incar = mock()
        incar.write.assert_called_once_with("INCAR_TAG = True")                                                                                                   

def test_write_False_to_incar():
    mock = mock_open()
    with patch("ase.io.vasp_parsers.incar_writer.open", mock):
        write_incar({"INCAR_TAG": False})
        mock.assert_called_once_with("INCAR", "w")
        incar = mock()
        incar.write.assert_called_once_with("INCAR_TAG = False")                                                                                                    
