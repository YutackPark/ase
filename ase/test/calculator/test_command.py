import subprocess

import pytest

from ase import Atoms

# case 1: nothing specified
# case 2: command specified via environment
# case 3: command specified via keyword


class InterceptedCommand(BaseException):
    def __init__(self, command):
        self.command = command


def mock_popen(command, shell=False, cwd=None, **kwargs):
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
    'gamess_us': {},
    'gaussian': {},
    'gromacs': {},
    'gulp': {},
    'mopac': {},
    'nwchem': {},
    'onetep': {},
    'openmx': dict(data_path='.', dft_data_year='13'),
    'psi4': {},
    'qchem': {},
    'siesta': dict(pseudo_path='.'),
    'turbomole': {},
    'vasp': {},
}


@pytest.fixture(autouse=True)
def miscellaneous_hacks(monkeypatch, tmp_path):
    from ase.calculators.calculator import FileIOCalculator
    from ase.calculators.demon import Demon
    from ase.calculators.crystal import CRYSTAL
    from ase.calculators.gamess_us import GAMESSUS
    from ase.calculators.gulp import GULP
    from ase.calculators.openmx import OpenMX
    from ase.calculators.siesta import Siesta
    from ase.calculators.vasp import Vasp

    def do_nothing(returnval=None):
        def mock_function(*args, **kwargs):
            return returnval
        return mock_function

    # Monkeypatches can be pretty dangerous because someone might obtain
    # a reference to the monkeypatched value before the patch is undone.
    #
    # We should try to refactor so we can avoid all the monkeypatches.

    monkeypatch.setattr(Demon, 'link_file', do_nothing())
    monkeypatch.setattr(CRYSTAL, '_write_crystal_in', do_nothing())

    # It calls super, but we'd like to skip the userscr handling:
    monkeypatch.setattr(GAMESSUS, 'calculate', FileIOCalculator.calculate)
    monkeypatch.setattr(GULP, 'library_check', do_nothing())

    # Attempts to read too many files.
    monkeypatch.setattr(OpenMX, 'write_input', do_nothing())

    monkeypatch.setattr(Siesta, '_write_species', do_nothing())
    monkeypatch.setattr(Vasp, '_build_pp_list', do_nothing(returnval=[]))


def mkcalc(name, **kwargs):
    from ase.calculators.calculator import get_calculator_class
    cls = get_calculator_class(name)
    kwargs = {**calculators[name], **kwargs}
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


def intercept_command(name, **kwargs):
    atoms = Atoms('H', pbc=True)
    atoms.center(vacuum=3.0)
    atoms.calc = mkcalc(name, **kwargs)
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
    'openmx': 'ASE_OPENMX_COMMAND',  # fails in get_dft_data_year
    # 'psi4', <-- has command but is Calculator
    # 'qchem': 'ASE_QCHEM_COMMAND',  # ignores environment
    'siesta': 'ASE_SIESTA_COMMAND',
    # 'turbomole': turbomole is not really a calculator
    'vasp': 'ASE_VASP_COMMAND',
}


def get_expected_command(command, name, tmp_path):
    expected_command = command
    if name == 'castep':
        expected_command = f'{command} castep'  # crazy
    elif name == 'dmol':
        expected_command = f'{command} tmp > tmp.out'
    elif name == 'gromacs':
        expected_command = (
            f'{command} mdrun -s gromacs.tpr -o gromacs.trr '
            '-e gromacs.edr -g gromacs.log -c gromacs.g96  > MM.log 2>&1')
    elif name == 'openmx':
        # openmx converts the stream target to an abspath, so the command
        # will vary depending on the tempdir we're running in.
        expected_command = f'{command} openmx.dat > {tmp_path}/openmx.log'
    return expected_command


@pytest.mark.parametrize('name', list(envvars))
def test_envvar(monkeypatch, name, tmp_path):
    command = 'dummy shell command from environment'
    expected_command = get_expected_command(command, name, tmp_path)
    monkeypatch.setenv(envvars[name], command)
    assert intercept_command(name) == expected_command


def keyword_calculator_list():
    skipped = {
        'turbomole',  # commands are hardcoded in turbomole
        'qchem',  # qchem does something entirely different.  wth
        'castep',  # has castep_command keyword instead
        'psi4',  # needs external package
        'onetep',  # ?
        'dmol',  # fixme
        'demon',  # fixme
    }
    return sorted(set(calculators) - skipped)


command_keywords = {'castep': 'CASTEP_COMMAND'}


@pytest.mark.parametrize('name', keyword_calculator_list())
def test_command_keyword(name, tmp_path):
    command = 'dummy command via keyword'
    expected_command = get_expected_command(command, name, tmp_path)

    # normally {'command': command}
    commandkwarg = {command_keywords.get(name, 'command'): command}
    assert intercept_command(name, **commandkwarg) == expected_command
