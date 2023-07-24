"""Test write and read."""
import io

import pytest
from ase import Atoms
from ase.build import bulk
from ase.data import atomic_numbers
from ase.io.lammpsdata import read_lammps_data, write_lammps_data


@pytest.mark.parametrize("masses", [False, True])
class _Base:
    def _run(self, atoms_ref: Atoms, masses: bool):
        buf = io.StringIO()
        write_lammps_data(buf, atoms_ref, masses=masses)
        buf.seek(0)
        # By default, write_lammps_data assigns atom types to the elements in
        # alphabetical order. To be consistent, here spiecies are also sorted.
        species = sorted(set(atoms_ref.get_chemical_symbols()))
        Z_of_type = {i + 1: atomic_numbers[s] for i, s in enumerate(species)}
        atoms = read_lammps_data(buf, Z_of_type=Z_of_type, style="atomic")
        self._compare(atoms, atoms_ref)

    def _compare(self, atoms: Atoms, atoms_ref: Atoms):
        assert all(atoms.numbers == atoms_ref.numbers)
        assert atoms.get_masses() == pytest.approx(atoms_ref.get_masses())
        assert atoms.get_scaled_positions() == pytest.approx(
            atoms_ref.get_scaled_positions())


@pytest.mark.parametrize("cubic", [False, True])
class TestCubic(_Base):
    """Test cubic structures."""

    def test_bcc(self, cubic: bool, masses: bool):
        """Test bcc."""
        atoms_ref = bulk("Li", "bcc", cubic=cubic)
        self._run(atoms_ref, masses)

    def test_fcc(self, cubic: bool, masses: bool):
        """Test fcc."""
        atoms_ref = bulk("Cu", "fcc", cubic=cubic)
        self._run(atoms_ref, masses)

    def test_rocksalt(self, cubic: bool, masses: bool):
        """Test rocksalt."""
        atoms_ref = bulk("NaCl", "rocksalt", a=1.0, cubic=cubic)
        self._run(atoms_ref, masses)

    def test_fluorite(self, cubic: bool, masses: bool):
        """Test fluorite."""
        atoms_ref = bulk("CaF2", "fluorite", a=1.0, cubic=cubic)
        self._run(atoms_ref, masses)


@pytest.mark.parametrize("orthorhombic", [False, True])
class TestOrthorhombic(_Base):
    """Test orthorhombic structures."""

    def test_hcp(self, masses: bool, orthorhombic: bool):
        """Test hcp."""
        atoms_ref = bulk("Mg", "hcp", orthorhombic=orthorhombic)
        self._run(atoms_ref, masses)
