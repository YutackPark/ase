"""Tests for FixCom."""
import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.constraints import FixCom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS


@pytest.fixture(name="atoms")
def fixture_atoms() -> Atoms:
    """fixture_atoms"""
    atoms = molecule('H2O')
    atoms.center(vacuum=4)
    atoms.calc = EMT()
    return atoms


def test_center_of_mass_position(atoms: Atoms):
    """Test if the center of mass does not move."""
    cold = atoms.get_center_of_mass()
    atoms.set_constraint(FixCom())

    with BFGS(atoms) as opt:
        opt.run(steps=5)

    cnew = atoms.get_center_of_mass()

    assert max(abs(cnew - cold)) < 1e-8


def test_center_of_mass_velocity(atoms: Atoms):
    """Test if the center-of-mass veloeicty is zero."""
    atoms.set_constraint(FixCom())

    # `adjust_momenta` of constaints are applied inside
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0)

    velocity_com = atoms.get_momenta().sum(axis=0) / atoms.get_masses().sum()

    assert max(abs(velocity_com)) < 1e-8
