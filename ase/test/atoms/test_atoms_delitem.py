"""Tests for `Atoms.__delitem__`"""
from ase.build import bulk
from ase.calculators.emt import EMT


def test_reset_calc_results():
    """Test if `Atoms.__delitem__` resets results in the attached calculator"""
    atoms = bulk('Cu', cubic=True)
    atoms.calc = EMT()
    atoms.get_potential_energy()
    del atoms[0]
    assert atoms.calc.results == {}
