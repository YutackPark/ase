import pytest

from ase.optimize import BFGS
from ase.calculators.emt import EMT


@pytest.fixture
def opt():
    from ase.build import bulk
    atoms = bulk('Au', cubic=True)
    atoms.rattle(stdev=0.12345, seed=42)
    atoms.calc = EMT()
    return BFGS(atoms)


# It is also a little bit questionable whether there should be
# three steps when we set steps=1.
@pytest.mark.parametrize('steps', [0, 1, 4])
def test_nsteps(opt, steps):
    irun = opt.irun(fmax=0, steps=steps)

    for i in range(steps + 2):
        next(irun)

    with pytest.raises(StopIteration):
        next(irun)
