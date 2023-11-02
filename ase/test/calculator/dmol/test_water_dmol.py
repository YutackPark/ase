from ase.build import molecule
from ase.calculators.dmol import DMol3


def test_water_dmol():

    atoms = molecule('H2O')
    calc = DMol3()
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.get_forces()
