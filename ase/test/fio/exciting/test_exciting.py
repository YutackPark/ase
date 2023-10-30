"""Test file for exciting file input and output methods."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import ase
import ase.io.exciting
# Import a realistic looking exciting text output file as a string.
from ase.test.calculator.exciting.test_exciting import LDA_VWN_AR_INFO_OUT


@pytest.fixture
def excitingtools():
    """If we cannot import excitingtools we skip tests with this fixture."""
    return pytest.importorskip('excitingtools')


@pytest.fixture
def nitrogen_trioxide_atoms():
    """Helper fixture to create a NO3 ase atoms object for tests."""
    return ase.Atoms('NO3',
                     cell=[[2, 2, 0], [0, 4, 0], [0, 0, 6]],
                     positions=[(0, 0, 0), (1, 3, 0),
                                (0, 0, 1), (0.5, 0.5, 0.5)],
                     pbc=True)


def test_write_input_xml_file(
        tmp_path, nitrogen_trioxide_atoms, excitingtools):
    """Test writing input.xml file using write_input_xml_file()."""
    file_path = tmp_path / 'input.xml'
    input_param_dict = {
        "rgkmax": 8.0,
        "do": "fromscratch",
        "ngridk": [6, 6, 6],
        "xctype": "GGA_PBE_SOL",
        "vkloff": [0, 0, 0],
        "tforce": True,
        "nosource": False
    }
    ase.io.exciting.write_input_xml_file(
        file_name=file_path,
        atoms=nitrogen_trioxide_atoms,
        input_parameters=input_param_dict,
        species_path="/dummy/arbitrary/path",
        title=None)
    assert file_path.exists()
    # Now read the XML file and ensure that it has what we expect:
    atoms_obj = ase.io.exciting.ase_atoms_from_exciting_input_xml(file_path)

    assert all(atoms_obj.symbols == "NOOO")
    input_xml_tree = ET.parse(file_path).getroot()
    parsed_calc_params = list(input_xml_tree)[2]
    assert parsed_calc_params.get("xctype") == "GGA_PBE_SOL"
    assert parsed_calc_params.get("rgkmax") == '8.0'
    assert parsed_calc_params.get("tforce") == 'true'


def test_ase_atoms_from_exciting_input_xml(
        tmp_path, nitrogen_trioxide_atoms, excitingtools):
    """Test reading the of the exciting input.xml file into ASE atoms obj."""
    expected_cell = np.array([[2, 2, 0], [0, 4, 0], [0, 0, 6]])
    expected_positions = np.array([(0, 0, 0), (1, 3, 0), (0, 0, 1),
                                   (0.5, 0.5, 0.5)])
    # First we write an input.xml file into a temp dir, so we can
    # read it back with our method we put under test.
    file_path = tmp_path / 'input.xml'
    input_param_dict = {
        "rgkmax": 8.0,
        "do": "fromscratch",
        "ngridk": [6, 6, 6],
        "xctype": "GGA_PBE_SOL",
        "vkloff": [0, 0, 0],
        "tforce": True,
        "nosource": False
    }
    ase.io.exciting.write_input_xml_file(
        file_name=file_path,
        atoms=nitrogen_trioxide_atoms,
        input_parameters=input_param_dict,
        species_path="/dummy/arbitrary/path",
        title=None)
    atoms_obj = ase.io.exciting.ase_atoms_from_exciting_input_xml(file_path)

    assert all(atoms_obj.symbols == "NOOO")
    assert atoms_obj.cell.array == pytest.approx(expected_cell)
    assert atoms_obj.positions == pytest.approx(expected_positions)


def test_parse_info_out_xml_bad_path(tmp_path, excitingtools):
    """Tests parse method raises error when info.out file doesn't exist."""
    output_file_path = Path(tmp_path).joinpath('info.out')

    with pytest.raises(FileNotFoundError):
        ase.io.exciting.parse_output(
            output_file_path)


def test_parse_info_out_energy(tmp_path, excitingtools):
    """Test parsing the INFO.OUT output from exciting using parse_output()."""
    expected_lattice_cell = [
        ['10.3360193975', '10.3426010725', '0.0054547264'],
        ['-10.3461511392', '10.3527307290', '0.0059928210'],
        ['10.3354645037', '10.3540072605', '20.6246241525']]

    file = tmp_path / "INFO.OUT"
    file.write_text(LDA_VWN_AR_INFO_OUT)
    assert file.exists(), "INFO.OUT written to tmp_path"

    results = ase.io.exciting.parse_output(file)

    # Finally ensure that we that the final SCL cycle is what we expect and
    # the final SCL results can be accessed correctly:
    final_scl_iteration = list(results["scl"].keys())[-1]
    assert pytest.approx(
        float(results["scl"][final_scl_iteration][
            "Hartree energy"])) == 205.65454603
    assert pytest.approx(
        float(results["scl"][final_scl_iteration][
            "Estimated fundamental gap"])) == 0.36095838
    assert pytest.approx(float(results["scl"][
        final_scl_iteration]["Hartree energy"])) == 205.65454603
    assert pytest.approx(float(
        results['initialization']['Unit cell volume'])) == 4412.7512103067
    assert results['initialization']['Total number of k-points'] == '1'
    assert results['initialization']['Maximum number of plane-waves'] == '251'
    # Grab the lattice vectors. excitingtools parses them in a fortran like
    # vector. We reshape accordingly into a 3x3 matrix where rows correspond
    # to lattice vectors.
    lattice_vectors_as_matrix = np.reshape(
        results['initialization']['Lattice vectors (cartesian)'],
        (3, 3), 'F')
    assert lattice_vectors_as_matrix.tolist() == expected_lattice_cell
