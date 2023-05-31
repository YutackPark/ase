"""Test file for exciting file input and output methods."""

from pathlib import Path

import pytest
import numpy as np
import xml.etree.ElementTree as ET

import ase
import ase.io.exciting
from ase.units import Bohr
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


def structure_xml_to_ase_atoms(file_path):
    """Helper function to parse the ase atoms data from the XML.

    This function is very simple and is used to verify the parsing that
    occurs with the excitingtools is working properly for simple systems.
    """
    # Parse file into element tree
    doc = ET.parse(file_path)
    root = doc.getroot()
    species_nodes = root.find('structure').iter('species')  # type: ignore

    symbols = []
    positions = []
    parsed_base_vectors = []

    # Collect data from tree
    for species_node in species_nodes:
        symbol = species_node.get('speciesfile').split('.')[0]  # type: ignore
        natoms = species_node.iter('atom')
        for atom in natoms:
            x_pos, y_pos, z_pos = atom.get('coord').split()  # type: ignore
            positions.append([float(x_pos), float(y_pos), float(z_pos)])
            symbols.append(symbol)

    # scale unit cell according to scaling attributes
    if 'scale' in doc.find('structure/crystal').attrib:  # type: ignore
        scale = float(str(
            doc.find('structure/crystal').attrib['scale']))  # type: ignore
    else:
        scale = 1

    if 'stretch' in doc.find('structure/crystal').attrib:  # type: ignore
        a_stretch, b_stretch, c_stretch = doc.find(  # type: ignore
            'structure/crystal').attrib['stretch'].text.split()
        stretch = np.array(
            [float(a_stretch), float(b_stretch), float(c_stretch)])
    else:
        stretch = np.array([1.0, 1.0, 1.0])

    raw_base_vectors = root.findall('structure/crystal/basevect')
    for base_vector in raw_base_vectors:
        x_mag, y_mag, z_mag = base_vector.text.split()  # type: ignore
        parsed_base_vectors.append(np.array([float(x_mag) * Bohr * stretch[0],
                                   float(y_mag) * Bohr * stretch[1],
                                   float(z_mag) * Bohr * stretch[2]
                                   ]) * scale)  # type: ignore
    atoms = ase.Atoms(symbols=symbols, cell=parsed_base_vectors)
    atoms.set_scaled_positions(positions)
    atoms.set_pbc(True)

    return atoms


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
        species_path=("/dummy/arbitrary/path"),
        title=None)
    assert file_path.exists()
    # Now read the XML file and ensure that it has what we expect:
    atoms_obj = structure_xml_to_ase_atoms(file_path)

    assert all(atoms_obj.symbols == "NOOO")
    input_xml_tree = ET.parse(file_path).getroot()
    parsed_calc_params = list(input_xml_tree)[2]
    assert parsed_calc_params.get("xctype") == "GGA_PBE_SOL"
    assert parsed_calc_params.get("rgkmax") == '8.0'
    assert parsed_calc_params.get("tforce") == 'true'


def test_ase_atoms_from_exciting_input_xml(
        tmp_path, nitrogen_trioxide_atoms, excitingtools):
    """Test reading the of the exciting input.xml file into ASE atoms obj."""
    expected_cell = [[2, 2, 0], [0, 4, 0], [0, 0, 6]]
    expected_positions = [(0, 0, 0), (1, 3, 0), (0, 0, 1), (0.5, 0.5, 0.5)]
    # First we write a an input.xml file into the a temp dir so we can
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
        species_path=("/dummy/arbitrary/path"),
        title=None)
    atoms_obj = ase.io.exciting.ase_atoms_from_exciting_input_xml(file_path)
    assert all(atoms_obj.symbols == "NOOO")
    # Convert numpy array's to lists to compare equality between arrays in
    # pytest.
    for i in range(np.shape(atoms_obj.cell)[1]):
        assert list(atoms_obj.cell[i]) == list(expected_cell[i])
    for j in range(len(expected_positions)):
        expected_positions[j] == atoms_obj.get_positions()[j]


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
