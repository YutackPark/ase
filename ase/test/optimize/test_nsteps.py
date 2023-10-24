import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import BFGS


@pytest.fixture
def opt():
    atoms = bulk('Au', cubic=True)
    atoms.rattle(stdev=0.12345, seed=42)
    atoms.calc = EMT()
    with BFGS(atoms) as opt:
        yield opt


# It is a little bit questionable whether there should be
# three steps when we set steps=1.
@pytest.mark.parametrize('steps', [0, 1, 4])
def test_nsteps(opt, steps):
    irun = opt.irun(fmax=0, steps=steps)

    for i in range(steps + 2):
        next(irun)

    with pytest.raises(StopIteration):
        next(irun)
