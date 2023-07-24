import pytest
import numpy as np

from ase.build import bulk

from ase.optimize import BFGS
from ase.calculators.socketio import SocketIOCalculator
from ase.constraints import ExpCellFilter
from ase.units import Ry


abinit_boilerplate = dict(
    ionmov=28,
    expert_user=1,
    optcell=2,
    tolmxf=1e-300,
    ntime=100_000,
    ecutsm=0.5,
)


@pytest.mark.calculator_lite
@pytest.mark.calculator('espresso', ecutwfc=200 / Ry)
@pytest.mark.calculator('abinit', ecut=200, **abinit_boilerplate)
def test_socketio_espresso(factory):
    atoms = bulk('Si')
    espresso = factory.calc(kpts=[2, 2, 2])
    atoms.rattle(stdev=.2, seed=42)

    with BFGS(ExpCellFilter(atoms)) as opt, \
            pytest.warns(UserWarning, match='Subprocess exited'), \
            SocketIOCalculator(
                espresso,
                unixsocket=f'ase_test_socketio_{factory.name}') as calc:
        atoms.calc = calc
        for _ in opt.irun(fmax=0.05):
            e = atoms.get_potential_energy()
            fmax = max(np.linalg.norm(atoms.get_forces(), axis=0))
            print(e, fmax)
