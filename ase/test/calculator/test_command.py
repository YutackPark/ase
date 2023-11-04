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


def mock_popen(command, shell, cwd):
    assert shell
    raise InterceptedCommand(command)


calculators = {
    'demonnano': dict(input_arguments={},
                      basis_path='hello'),
    'demon': dict(basis_path='hello'),
}


#class Factory:
#    def __init__(self, name, kwargs):
#        self.name = name
#        self.kwargs = kwargs

#    def calc(self):
@pytest.fixture(autouse=True)
def miscellaneous_hacks(monkeypatch):
    from ase.calculators.demon import Demon

    monkeypatch.setattr(Demon, 'link_file', lambda *args, **kwargs: None)


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
    'demonnano': 'ASE_DEMONNANO_COMMAND',
}

@pytest.mark.parametrize('name', ['demonnano'])
def test_envvar(monkeypatch, name):
    command = 'dummy shell command from environment'
    monkeypatch.setenv(envvars[name], command)
    assert intercept_command(name) == command


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
