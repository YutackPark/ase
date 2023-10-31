import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.cluster import Icosahedron
from ase.optimize import (BFGS, FIRE, LBFGS, Berny, BFGSLineSearch,
                          GoodOldQuasiNewton, GPMin, LBFGSLineSearch, MDMin,
                          ODE12r)
from ase.optimize.precon import PreconFIRE, PreconLBFGS, PreconODE12r
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG

optclasses = [
    MDMin,
    FIRE,
    LBFGS,
    LBFGSLineSearch,
    BFGSLineSearch,
    BFGS,
    GoodOldQuasiNewton,
    GPMin,
    SciPyFminCG,
    SciPyFminBFGS,
    PreconLBFGS,
    PreconFIRE,
    Berny,
    ODE12r,
    PreconODE12r,
]


@pytest.fixture(name="ref_atoms")
def fixture_ref_atoms(optcls):
    if optcls is Berny:
        ref_atoms = Icosahedron("Ag", 2, 3.82975)
        ref_atoms.calc = EMT()
        return ref_atoms

    ref_atoms = bulk("Au")
    ref_atoms.calc = EMT()
    ref_atoms.get_potential_energy()
    return ref_atoms


@pytest.fixture(name="atoms")
def fixture_atoms(ref_atoms, optcls):
    if optcls is Berny:
        atoms = ref_atoms.copy()
        floor = 7
    else:
        atoms = ref_atoms * (2, 2, 2)
        floor = 0.45

    atoms.calc = EMT()
    atoms.rattle(stdev=0.1, seed=7)
    e_unopt = atoms.get_potential_energy()
    assert e_unopt > floor
    return atoms


@pytest.fixture(name="optcls", params=optclasses)
def fixture_optcls(request):
    optcls = request.param
    if optcls is Berny:
        pytest.importorskip("berny")  # check if pyberny installed
    return optcls


@pytest.fixture(name="kwargs")
def fixture_kwargs(optcls):
    kwargs = {}
    if optcls is PreconLBFGS:
        kwargs["precon"] = None
    if optcls is Berny:
        kwargs["dihedral"] = False
    yield kwargs
    kwargs = {}


@pytest.mark.optimize
@pytest.mark.filterwarnings("ignore: estimate_mu")
def test_optimize(optcls, atoms, ref_atoms, kwargs):
    """Test if forces can be converged using the optimizer."""
    fmax = 0.01
    with optcls(atoms, **kwargs) as opt:
        is_converged = opt.run(fmax=fmax)
    assert is_converged  # check if opt.run() returns True when converged

    forces = atoms.get_forces()
    final_fmax = max((forces**2).sum(axis=1) ** 0.5)
    ref_energy = ref_atoms.get_potential_energy()
    e_opt = atoms.get_potential_energy() * len(ref_atoms) / len(atoms)
    e_err = abs(e_opt - ref_energy)

    print(f"{optcls.__name__:>20}:", end=" ")
    print(f"fmax={final_fmax:.05f} eopt={e_opt:.06f} err={e_err:06e}")

    assert final_fmax < fmax
    assert e_err < 1.75e-5  # (This tolerance is arbitrary)


@pytest.mark.optimize
def test_unconverged(optcls, atoms, kwargs):
    """Test if things work properly when forces are not converged."""
    fmax = 1e-9  # small value to not get converged
    with optcls(atoms, **kwargs) as opt:
        opt.run(fmax=fmax, steps=1)  # only one step to not get converged
    assert not opt.converged()
