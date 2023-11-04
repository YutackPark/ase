import pytest

from ase import Atoms
#from ase.calculators.gaussian import Gaussian
from ase.calculators.demonnano import DemonNano
from ase.calculators.names import names as calculator_names
import subprocess

# gaussian has weird handling


# case 1: nothing specified
# case 2: command specified via environment
# case 3: command specified via keyword


#names = ['ace', 'amber', 'castep',
#         'crystal', 'dmol',
#         'elk', 'espresso', 'exciting', 'gamess_us',
#         'gaussian', 'gromacs', 'gulp',
#         'mopac', 'morse', 'nwchem',
#         'onetep', 'openmx', 'orca',
#         'plumed', 'psi4', 'qchem', 'siesta',
#         'turbomole', 'vasp']


class InterceptedCommand(BaseException):
    def __init__(self, command):
        self.command = command


def mock_popen(command, shell, cwd, **kwargs):
    # castep passes stdout/stderr
    assert shell
    raise InterceptedCommand(command)


calculators = {
    'ace': {},
    'amber': {},
    'castep': dict(keyword_tolerance=3),
    'crystal': {},
    'demon': dict(basis_path='hello'),
    'demonnano': dict(input_arguments={},
                      basis_path='hello'),
    'dmol': {},
    'elk': {},
    'espresso': {},
    'exciting': {},
    'gamess_us': {},
    'gaussian': {},
    'gromacs': {},
    'gulp': {},
    'mopac': {},
    'morse': {},
    'nwchem': {},
    'onetep': {},
    'openmx': {},
    'orca': {},
    'plumed': {},
    'psi4': {},
    'qchem': {},
    'siesta': {},
    'turbomole': {},
    'vasp': {},
}


@pytest.fixture(autouse=True)
def miscellaneous_hacks(monkeypatch):
    from ase.calculators.demon import Demon
    from ase.calculators.castep import Castep
    from ase.calculators.crystal import CRYSTAL

    def do_nothing(retval=None):
        def mock_function(*args, **kwargs):
            return retval
        return mock_function

    monkeypatch.setattr(Demon, 'link_file', do_nothing())
    monkeypatch.setattr(CRYSTAL, '_write_crystal_in', do_nothing())
    #monkeypatch.setattr(Castep, 'import_castep_keywords',
    #                    lambda *args, **kwargs: {})

#@pytest.fixture(params=list(calculators))
def mkcalc(name):
    from ase.calculators.calculator import get_calculator_class
    cls = get_calculator_class(name)
    kwargs = calculators[name]
    return cls(**kwargs)


@pytest.mark.parametrize('name', ['demon', 'demonnano'])
def test_default(name):
    from ase.calculators.calculator import CalculatorSetupError
    with pytest.raises(CalculatorSetupError):
        intercept_command(name)


@pytest.fixture(autouse=True)
def mock_subprocess_popen(monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)


def intercept_command(name):
    atoms = Atoms('H')
    atoms.center(vacuum=3.0)
    atoms.calc = mkcalc(name)
    try:
        atoms.get_potential_energy()
    except InterceptedCommand as err:
        return err.command


envvars = {
    'ace': 'ASE_ACE_COMMAND',
    'amber': 'ASE_AMBER_COMMAND',
    'castep': 'CASTEP_COMMAND',
    'crystal': 'ASE_CRYSTAL_COMMAND',
    'demon': 'ASE_DEMON_COMMAND',
    'demonnano': 'ASE_DEMONNANO_COMMAND',
    'dmol': 'DMOL_COMMAND',  # XXX
}


#names = [
#         'elk', 'espresso', 'exciting', 'gamess_us',
#         'gaussian', 'gromacs', 'gulp',
#         'mopac', 'morse', 'nwchem',
#         'onetep', 'openmx', 'orca',
#         'plumed', 'psi4', 'qchem', 'siesta',
#         'turbomole', 'vasp']

@pytest.mark.parametrize('name', list(envvars))
def test_envvar(monkeypatch, name):
    command = 'dummy shell command from environment'
    expected_command = command
    if name == 'castep':
        expected_command = f'{command} castep'  # crazy
    elif name == 'dmol':
        expected_command = f'{command} tmp > tmp.out'
    monkeypatch.setenv(envvars[name], command)
    assert intercept_command(name) == expected_command


#@pytest.mark.parametrize('kind', ['default', 'command', 'envvar'])
def t1est_command(monkeypatch, kind, calcname):
    atoms = Atoms('H')
    atoms.center(vacuum=3.0)

    dummy_command = 'hello world'

    command = None
    if kind == 'command':
        command = dummy_command
        kwargs = {**kwargs, 'command': dummy_command}
    elif kind == 'envvar':
        monkeypatch.setenv('ASE_DEMONNANO_COMMAND', dummy_command)
    else:
        assert kind == 'default'

    atoms.calc = calc #DemonNano(basis_path='hello',
                           # input_arguments={},
                           # command=dummy_command
                 #          )

    # monkeypatch.setattr(subprocess, 'Popen', mock_popen)
    try:
        atoms.get_potential_energy()
    except InterceptedCommand as err:
        command = err.command

    assert command == dummy_command
