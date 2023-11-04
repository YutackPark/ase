import subprocess

import pytest

from ase import Atoms

# gaussian has weird handling


# case 1: nothing specified
# case 2: command specified via environment
# case 3: command specified via keyword


# names = ['ace', 'amber', 'castep',
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


def mock_popen(command, shell=False, cwd=None, **kwargs):
    # castep passes stdout/stderr
    assert shell
    raise InterceptedCommand(command)


# Parameters for each calculator -- whatever it takes trigger a calculation
# without crashing first.
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
    'openmx': dict(data_path='.'),
    'plumed': {},
    'psi4': {},
    'qchem': {},
    'siesta': dict(pseudo_path='.'),
    'turbomole': {},
    'vasp': {},
}


@pytest.fixture(autouse=True)
def miscellaneous_hacks(monkeypatch, tmp_path):
    from ase.calculators.demon import Demon
    from ase.calculators.crystal import CRYSTAL
    from ase.calculators.gamess_us import GAMESSUS
    from ase.calculators.gulp import GULP
    from ase.calculators.calculator import FileIOCalculator
    from ase.calculators.siesta import Siesta
    from ase.calculators.vasp import Vasp

    def do_nothing(returnval=None):
        def mock_function(*args, **kwargs):
            return returnval
        return mock_function

    monkeypatch.setattr(Demon, 'link_file', do_nothing())
    monkeypatch.setattr(CRYSTAL, '_write_crystal_in', do_nothing())

    # It calls super, but we'd like to skip the userscr handling:
    monkeypatch.setattr(GAMESSUS, 'calculate', FileIOCalculator.calculate)
    monkeypatch.setattr(GULP, 'library_check', do_nothing())

    monkeypatch.setattr(Siesta, '_write_species', do_nothing())
    monkeypatch.setattr(Vasp, '_build_pp_list', do_nothing(returnval=[]))


def mkcalc(name):
    from ase.calculators.calculator import get_calculator_class
    cls = get_calculator_class(name)
    kwargs = calculators[name]
    return cls(**kwargs)


@pytest.mark.parametrize('name', ['demon', 'demonnano'])
def test_default(name, monkeypatch):
    from ase.calculators.calculator import CalculatorSetupError

    # Make sure it does not pickup system var we don't know about:
    if name in envvars:
        monkeypatch.delenv(envvars[name], raising=False)

    with pytest.raises(CalculatorSetupError):
        intercept_command(name)


@pytest.fixture(autouse=True)
def mock_subprocess_popen(monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)


def intercept_command(name):
    atoms = Atoms('H', pbc=True)
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
    'dmol': 'DMOL_COMMAND',  # XXX Crashes when it runs along other tests
    'elk': 'ASE_ELK_COMMAND',
    'gamess_us': 'ASE_GAMESSUS_COMMAND',
    'gaussian': 'ASE_GAUSSIAN_COMMAND',
    'gromacs': 'ASE_GROMACS_COMMAND',
    'gulp': 'ASE_GULP_COMMAND',
    'mopac': 'ASE_MOPAC_COMMAND',
    'nwchem': 'ASE_NWCHEM_COMMAND',
    # 'openmx': 'ASE_OPENMX_COMMAND',  # fails in get_dft_data_year
    # 'psi4', <-- has command but is Calculator
    # 'qchem': 'ASE_QCHEM_COMMAND',  # ignores environment
    'siesta': 'ASE_SIESTA_COMMAND',
    # 'turbomole': turbomole is not really a calculator
    'vasp': 'ASE_VASP_COMMAND',
}


@pytest.mark.parametrize('name', list(envvars))
def test_envvar(monkeypatch, name):
    command = 'dummy shell command from environment'
    expected_command = command
    if name == 'castep':
        expected_command = f'{command} castep'  # crazy
    elif name == 'dmol':
        expected_command = f'{command} tmp > tmp.out'
    elif name == 'gromacs':
        expected_command = (
            f'{command} mdrun -s gromacs.tpr -o gromacs.trr '
            '-e gromacs.edr -g gromacs.log -c gromacs.g96  > MM.log 2>&1')

    monkeypatch.setenv(envvars[name], command)
    assert intercept_command(name) == expected_command
