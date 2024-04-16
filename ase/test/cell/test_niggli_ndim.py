import numpy as np
import pytest

from ase.cell import Cell


def test_niggli_0d():
    rcell, op = Cell.new().niggli_reduce()
    assert rcell.rank == 0
    assert (op == np.identity(3, dtype=int)).all()


def test_niggli_1d():
    cell = Cell.new()
    vector = [1, 2, 3]
    cell[1] = vector

    rcell, op = cell.niggli_reduce()
    assert rcell.lengths() == pytest.approx([np.linalg.norm(vector), 0, 0])
    assert Cell(op.T @ cell).cellpar() == pytest.approx(rcell.cellpar())


def test_niggli_2d():
    cell = Cell.new()
    cell[0] = [3, 4, 5]
    cell[2] = [5, 6, 7]
    rcell, op = cell.niggli_reduce()
    assert rcell.rank == 2
    assert rcell.lengths()[2] == 0
    assert Cell(op.T @ cell).cellpar() == pytest.approx(rcell.cellpar())


def test_niggli_2d_atoms():
    from ase.build import fcc111
    from ase.build import niggli_reduce
    from ase.calculators.emt import EMT
    from ase.visualize import view

    atoms = fcc111('Au', (2, 2, 1), vacuum=2.0)
    atoms.cell[2] = [0, 0, 1e-3]
    atoms.rattle(stdev=0.1)
    print()
    print(atoms.cell[:])
    # xxx
    atoms.calc = EMT()
    atoms1 = atoms.copy()
    e1 = atoms.get_potential_energy()
    niggli_reduce(atoms)
    e2 = atoms.get_potential_energy()

    # view([atoms1, atoms])

    assert e2 == pytest.approx(e1, abs=1e-10)
