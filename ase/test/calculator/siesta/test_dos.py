import pytest

from ase.dft import DOS
from ase.build import bulk


@pytest.mark.calculator_lite
@pytest.mark.calculator('siesta')
def test_dos(factory):
    atoms = bulk('Si')
    atoms.calc = factory.calc(kpts=[2, 2, 2])
    atoms.get_potential_energy()
    DOS(atoms.calc, width=0.2)
