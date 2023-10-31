from ase.io.vasp_parsers.incar_writer import write_incar
from unittest.mock import mock_open, patch


def test_write_string_to_incar():
    mock = mock_open()
    with patch("ase.io.vasp_parsers.incar_writer.open", mock):
        write_incar({"INCAR_TAG": "string"})
        mock.assert_called_once_with("INCAR", "w")
        incar = mock()
        incar.write.assert_called_once_with("INCAR_TAG = string")
