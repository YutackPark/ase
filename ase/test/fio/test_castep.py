import re
import warnings

import ase.build
import numpy as np
import pytest
from ase.io import read, write


# create mol with custom mass - from a list of positions or using
# ase.build.molecule
def write_read_atoms(atom, tmp_path):
    write("{0}/{1}".format(tmp_path, "castep_test.cell"), atom)
    return read("{0}/{1}".format(tmp_path, "castep_test.cell"))


# write to .cell and check that .cell has correct species_mass block in it
@pytest.mark.parametrize(
    "mol, custom_masses, expected_species, expected_mass_block",
    [
        ("CH4", {2: [1]}, ["C", "H:0", "H", "H", "H"], ["H:0 2.0"]),
        ("CH4", {2: [1, 2, 3, 4]}, ["C", "H", "H", "H", "H"], ["H 2.0"]),
        ("C2H5", {2: [2, 3]}, ["C", "C", "H:0",
         "H:0", "H", "H", "H"], ["H:0 2.0"]),
        (
            "C2H5",
            {2: [2], 3: [3]},
            ["C", "C", "H:0", "H:1", "H", "H", "H"],
            ["H:0 2.0", "H:1 3.0"],
        ),
    ],
)
def test_custom_mass_write(
    mol, custom_masses, expected_species, expected_mass_block, tmp_path
):

    custom_atoms = ase.build.molecule(mol)
    atom_positions = custom_atoms.positions

    for mass, indices in custom_masses.items():
        for i in indices:
            custom_atoms[i].mass = mass

    atom_masses = custom_atoms.get_masses()
    # CASTEP IO can be noisy while handling keywords JSON
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        new_atoms = write_read_atoms(custom_atoms, tmp_path)

    # check atoms have been written and read correctly
    np.testing.assert_allclose(atom_positions, new_atoms.positions)
    np.testing.assert_allclose(atom_masses, new_atoms.get_masses())

    # check that file contains appropriate blocks
    with open("{0}/{1}".format(tmp_path, "castep_test.cell"), "r") as f:
        data = f.read().replace("\n", "\\n")

    position_block = re.search(
        r"%BLOCK POSITIONS_ABS.*%ENDBLOCK POSITIONS_ABS", data)
    assert position_block

    pos = position_block.group().split("\\n")[1:-1]
    species = [p.split(" ")[0] for p in pos]
    assert species == expected_species

    mass_block = re.search(r"%BLOCK SPECIES_MASS.*%ENDBLOCK SPECIES_MASS", data)
    assert mass_block

    masses = mass_block.group().split("\\n")[1:-1]
    for line, expected_line in zip(masses, expected_mass_block):
        species_name, mass_read = line.split(' ')
        expected_species_name, expected_mass = expected_line.split(' ')
        assert pytest.approx(float(mass_read), abs=1e-6) == float(expected_mass)
        assert species_name == expected_species_name


# test setting a custom species on different atom before write
def test_custom_mass_overwrite(tmp_path):
    custom_atoms = ase.build.molecule("CH4")
    custom_atoms[1].mass = 2

    # CASTEP IO is noisy while handling keywords JSON
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        atoms = write_read_atoms(custom_atoms, tmp_path)

    # test that changing masses when custom masses defined causes errors
    atoms[3].mass = 3
    with pytest.raises(ValueError,
                       match="Could not write custom mass block for H."):
        atoms.write("{0}/{1}".format(tmp_path, "castep_test2.cell"))
