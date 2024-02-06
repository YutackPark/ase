import os
import re
from pathlib import Path
import tempfile

import importlib.util
import pytest

from ase import Atoms

from ase.calculators.calculator import get_calculator_class
from ase.calculators.names import (names as calculator_names,
                                   builtin)
from ase.calculators.genericfileio import read_stdout
from ase.config import Config
from ase.utils import lazyproperty


class NotInstalled(Exception):
    pass


class MachineInformation:
    @staticmethod
    def please_install_ase_datafiles():
        return ImportError("""\
Could not import asetest package.  Please install ase-datafiles
using e.g. "pip install ase-datafiles" to run calculator integration
tests.""")

    @lazyproperty
    def datafiles_module(self):
        try:
            import asetest
        except ModuleNotFoundError:
            return None
        return asetest

    @lazyproperty
    def datafile_config(self):
        # XXXX TODO avoid requiring the dummy [parallel] section
        datafiles = self.datafiles_module
        if datafiles is None:
            return ''  # empty configfile
        path = self.datafiles_module.paths.DataFiles().datapath
        datafile_config = f"""\
# Dummy parallel section (make this unnecessary)
[parallel]

# Configuration for ase-datafiles

[abinit]
pp_paths =
    {path}/abinit/GGA_FHI
    {path}/abinit/LDA_FHI
    {path}/abinit/LDA_PAW


[dftb]
skt_paths = {path}/dftb

[elk]
species_dir = {path}/elk

[espresso]
pseudo_dir = {path}/espresso/gbrv-lda-espresso

[lammps]
potentials = {path}/lammps

[openmx]
data_path = {path}/openmx

[siesta]
pseudo_path = {path}/siesta
"""
        return datafile_config

    @lazyproperty
    def cfg(self):
        # First we load the usual configfile.
        # But we don't want to run tests against the user's production
        # configuration since that may be using other pseudopotentials
        # than the ones we want.  Therefore, we override datafile paths.
        cfg = Config.read()
        cfg.parser.read_string(self.datafile_config)
        return cfg


factory_classes = {}


def factory(name):
    def decorator(cls):
        cls.name = name
        assert name not in factory_classes, name
        factory_classes[name] = cls
        return cls

    return decorator


def make_factory_fixture(name):
    @pytest.fixture(scope='session')
    def _factory(factories):
        factories.require(name)
        return factories[name]

    _factory.__name__ = f'{name}_factory'
    return _factory


@factory('abinit')
class AbinitFactory:
    def __init__(self, cfg):
        from ase.calculators.abinit import AbinitTemplate
        self._profile = AbinitTemplate().load_profile(cfg)

    def version(self):
        return self._profile.version()

    def _base_kw(self):
        return dict(ecut=150, chksymbreak=0, toldfe=1e-3)

    def calc(self, **kwargs):
        from ase.calculators.abinit import Abinit
        kwargs = {**self._base_kw(), **kwargs}
        return Abinit(profile=self._profile, **kwargs)

    def socketio(self, unixsocket, **kwargs):
        kwargs = {
            'tolmxf': 1e-300,
            'ntime': 100_000,
            'ecutsm': 0.5,
            'ecut': 200,
            **kwargs,
        }

        return self.calc(**kwargs).socketio(unixsocket=unixsocket)


@factory('aims')
class AimsFactory:
    def __init__(self, cfg):
        from ase.calculators.aims import AimsTemplate
        self._profile = AimsTemplate().load_profile(cfg)

    def calc(self, **kwargs):
        from ase.calculators.aims import Aims

        kwargs1 = dict(xc='LDA')
        kwargs1.update(kwargs)
        return Aims(profile=self._profile, **kwargs1)

    def version(self):
        return self._profile.version()

    def socketio(self, unixsocket, **kwargs):
        return self.calc(**kwargs).socketio(unixsocket=unixsocket)


@factory('asap')
class AsapFactory:
    importname = 'asap3'

    def __init__(self, cfg):
        spec = importlib.util.find_spec('asap3')
        if spec is None:
            raise NotInstalled('asap3')

    def _asap3(self):
        import asap3
        return asap3

    def calc(self, **kwargs):
        return self._asap3().EMT(**kwargs)

    def version(self):
        return self._asap3().__version__


@factory('cp2k')
class CP2KFactory:
    def __init__(self, cfg):
        self.executable = cfg.parser['cp2k']['cp2k_shell']

    def version(self):
        from ase.calculators.cp2k import Cp2kShell
        shell = Cp2kShell(self.executable, debug=False)
        return shell.version

    def calc(self, **kwargs):
        from ase.calculators.cp2k import CP2K
        return CP2K(command=self.executable, **kwargs)


@factory('castep')
class CastepFactory:
    def __init__(self, cfg):
        self.executable = cfg.parser['castep']['binary']

    def version(self):
        from ase.calculators.castep import get_castep_version

        return get_castep_version(self.executable)

    def calc(self, **kwargs):
        from ase.calculators.castep import Castep

        return Castep(castep_command=self.executable, **kwargs)


@factory('dftb')
class DFTBFactory:
    def __init__(self, cfg):
        self.executable = cfg.parser['dftb']['binary']
        self.skt_path = cfg.parser['dftb']['skt_paths']  # multiple paths?
        # assert len(self.skt_paths) == 1  # XXX instructive error?

    def version(self):
        stdout = read_stdout([self.executable])
        match = re.search(r'DFTB\+ release\s*(\S+)', stdout, re.M)
        return match.group(1)

    def calc(self, **kwargs):
        from ase.calculators.dftb import Dftb

        command = f'{self.executable} > PREFIX.out'
        return Dftb(
            command=command,
            slako_dir=str(self.skt_path) + '/',  # XXX not obvious
            **kwargs,
        )

    def socketio_kwargs(self, unixsocket):
        return dict(
            Driver_='', Driver_Socket_='', Driver_Socket_File=unixsocket
        )


@factory('dftd3')
class DFTD3Factory:
    def __init__(self, cfg):
        self.executable = cfg.parser['dftd3']['binary']

    def calc(self, **kwargs):
        from ase.calculators.dftd3 import DFTD3

        return DFTD3(command=self.executable, **kwargs)


@factory('elk')
class ElkFactory:
    def __init__(self, cfg):
        self.executable = cfg.parser['elk']['binary']
        self.species_dir = cfg.parser['elk']['species_dir']

    def version(self):
        output = read_stdout([self.executable])
        match = re.search(r'Elk code version (\S+)', output, re.M)
        return match.group(1)

    def calc(self, **kwargs):
        from ase.calculators.elk import ELK

        command = f'{self.executable} > elk.out'
        return ELK(command=command, species_dir=self.species_dir, **kwargs)


@factory('espresso')
class EspressoFactory:
    def __init__(self, cfg):
        from ase.calculators.espresso import EspressoTemplate
        self.profile = EspressoTemplate().load_profile(cfg)

    def version(self):
        return self.profile().version()

    @lazyproperty
    def pseudopotentials(self):
        pseudopotentials = {}
        for path in self.profile.pseudo_dir.glob('*.UPF'):
            fname = path.name
            # Names are e.g. si_lda_v1.uspp.F.UPF
            symbol = fname.split('_', 1)[0].capitalize()
            pseudopotentials[symbol] = fname
        return pseudopotentials

    def calc(self, **kwargs):
        from ase.calculators.espresso import Espresso
        from ase.io.espresso import Namelist

        input_data = Namelist(kwargs.pop("input_data", None))
        input_data.to_nested()
        input_data["system"].setdefault("ecutwfc", 22.05)

        return Espresso(
            profile=self.profile, pseudopotentials=self.pseudopotentials,
            input_data=input_data, **kwargs
        )

    def socketio(self, unixsocket, **kwargs):
        return self.calc(**kwargs).socketio(unixsocket=unixsocket)


@factory('exciting')
class ExcitingFactory:
    """Factory to run exciting tests."""

    def __init__(self, cfg):
        from ase.calculators.exciting.exciting import (
            ExcitingGroundStateTemplate)

        # Where do species come from?  We do not have them in ase-datafiles.
        # We should specify species_path.
        self._profile = ExcitingGroundStateTemplate().load_profile(cfg)

    def calc(self, **kwargs):
        """Get instance of Exciting Ground state calculator."""
        from ase.calculators.exciting.exciting import (
            ExcitingGroundStateCalculator)

        return ExcitingGroundStateCalculator(
            ground_state_input=kwargs, species_path=self._profile.species_path
        )

    def version(self):
        """Get exciting executable version."""
        return self._profile.version


@factory('mopac')
class MOPACFactory:
    def __init__(self, cfg):
        self.executable = cfg.parser['mopac']['binary']

    def calc(self, **kwargs):
        from ase.calculators.mopac import MOPAC

        command = f'{self.executable} PREFIX.mop 2> /dev/null'
        return MOPAC(command=command, **kwargs)

    def version(self):
        cwd = Path('.').absolute()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                h = Atoms('H')
                h.calc = self.calc()
                _ = h.get_potential_energy()
            finally:
                os.chdir(cwd)

        return h.calc.results['version']


@factory('vasp')
class VaspFactory:
    def __init__(self, cfg):
        self.executable = cfg.parser['vasp']['binary']

    def version(self):
        from ase.calculators.vasp import get_vasp_version

        header = read_stdout([self.executable], createfile='INCAR')
        return get_vasp_version(header)

    def calc(self, **kwargs):
        from ase.calculators.vasp import Vasp

        # XXX We assume the user has set VASP_PP_PATH
        if Vasp.VASP_PP_PATH not in os.environ:
            # For now, we skip with a message that we cannot run the test
            pytest.skip(
                'No VASP pseudopotential path set. '
                'Set the ${} environment variable to enable.'.format(
                    Vasp.VASP_PP_PATH
                )
            )
        return Vasp(command=self.executable, **kwargs)


@factory('gpaw')
class GPAWFactory:
    importname = 'gpaw'

    def __init__(self, cfg):
        spec = importlib.util.find_spec('gpaw')
        # XXX should be made non-pytest dependent
        if spec is None:
            raise NotInstalled('gpaw')

    def calc(self, **kwargs):
        from gpaw import GPAW

        return GPAW(**kwargs)

    def version(self):
        import gpaw

        return gpaw.__version__


@factory('psi4')
class Psi4Factory:
    importname = 'psi4'

    def __init__(self, cfg):
        try:
            import psi4  # noqa
        except ModuleNotFoundError:
            raise NotInstalled('psi4')

    def calc(self, **kwargs):
        from ase.calculators.psi4 import Psi4

        return Psi4(**kwargs)


@factory('gromacs')
class GromacsFactory:
    def __init__(self, cfg):
        self.executable = cfg.parser['gromacs']['binary']

    def version(self):
        from ase.calculators.gromacs import get_gromacs_version

        return get_gromacs_version(self.executable)

    def calc(self, **kwargs):
        from ase.calculators.gromacs import Gromacs

        return Gromacs(command=self.executable, **kwargs)


class BuiltinCalculatorFactory:
    def __init__(self, cfg):
        self.cfg = cfg

    def calc(self, **kwargs):
        cls = get_calculator_class(self.name)
        return cls(**kwargs)


@factory('eam')
class EAMFactory(BuiltinCalculatorFactory):
    def __init__(self, cfg):
        self.potentials_path = cfg.parser['lammps']['potentials']


@factory('emt')
class EMTFactory(BuiltinCalculatorFactory):
    pass


@factory('lammpsrun')
class LammpsRunFactory:
    def __init__(self, cfg):
        self.executable = cfg.parser['lammps']['binary']
        self.potentials_path = cfg.parser['lammps']['potentials']
        # XXX if lammps wants this variable set we should pass it to Popen.
        # But if ASE wants it, it should be passed programmatically.
        os.environ['LAMMPS_POTENTIALS'] = str(self.potentials_path)

    def version(self):
        stdout = read_stdout([self.executable])
        match = re.match(r'LAMMPS\s*\((.+?)\)', stdout, re.M)
        return match.group(1)

    def calc(self, **kwargs):
        from ase.calculators.lammpsrun import LAMMPS

        return LAMMPS(command=self.executable, **kwargs)


@factory('lammpslib')
class LammpsLibFactory:
    def __init__(self, cfg):
        # XXX FIXME need to stop gracefully if lammpslib python
        # package is not installed
        #
        # Set the path where LAMMPS will look for potential parameter files
        try:
            import lammps  # noqa: F401
        except ModuleNotFoundError:
            raise NotInstalled('missing python wrappers: cannot import lammps')
        self.potentials_path = cfg.parser['lammps']['potentials']
        os.environ['LAMMPS_POTENTIALS'] = str(self.potentials_path)

    def version(self):
        import lammps

        cmd_args = [
            '-echo',
            'log',
            '-log',
            'none',
            '-screen',
            'none',
            '-nocite',
        ]
        lmp = lammps.lammps(name='', cmdargs=cmd_args, comm=None)
        try:
            return lmp.version()
        finally:
            lmp.close()

    def calc(self, **kwargs):
        from ase.calculators.lammpslib import LAMMPSlib

        return LAMMPSlib(**kwargs)


@factory('openmx')
class OpenMXFactory:
    def __init__(self, cfg):
        # XXX Cannot test this, is surely broken.
        self.executable = cfg.parser['openmx']['binary']
        self.data_path = cfg.parser['openmx'][0]

    def version(self):
        from ase.calculators.openmx.openmx import parse_omx_version

        dummyfile = 'omx_dummy_input'
        stdout = read_stdout([self.executable, dummyfile], createfile=dummyfile)
        return parse_omx_version(stdout)

    def calc(self, **kwargs):
        from ase.calculators.openmx import OpenMX

        return OpenMX(
            command=self.executable, data_path=str(self.data_path), **kwargs
        )


@factory('octopus')
class OctopusFactory:
    def __init__(self, cfg):
        from ase.calculators.octopus import OctopusTemplate
        self._profile = OctopusTemplate().load_profile(cfg)

    def version(self):
        return self._profile.version()

    def calc(self, **kwargs):
        from ase.calculators.octopus import Octopus
        return Octopus(profile=self._profile, **kwargs)


@factory('orca')
class OrcaFactory:
    def __init__(self, cfg):
        self.executable = cfg.parser['orca']['binary']

    def _profile(self):
        from ase.calculators.orca import OrcaProfile

        return OrcaProfile(self.executable)

    def version(self):
        return self._profile().version()

    def calc(self, **kwargs):
        from ase.calculators.orca import ORCA

        return ORCA(**kwargs)


@factory('siesta')
class SiestaFactory:
    def __init__(self, cfg):
        self.executable = cfg.parser['siesta']['binary']
        self.pseudo_path = str(cfg.parser['siesta']['pseudo_path'])

    def version(self):
        from ase.calculators.siesta.siesta import get_siesta_version

        full_ver = get_siesta_version(self.executable)
        m = re.match(r'siesta-(\S+)', full_ver, flags=re.I)
        if m:
            return m.group(1)
        return full_ver

    def calc(self, **kwargs):
        from ase.calculators.siesta import Siesta

        command = f'{self.executable} < PREFIX.fdf > PREFIX.out'
        return Siesta(
            command=command, pseudo_path=str(self.pseudo_path), **kwargs
        )

    def socketio_kwargs(self, unixsocket):
        return {
            'fdf_arguments': {
                'MD.TypeOfRun': 'Master',
                'Master.code': 'i-pi',
                'Master.interface': 'socket',
                'Master.address': unixsocket,
                'Master.socketType': 'unix',
            }
        }


@factory('nwchem')
class NWChemFactory:
    def __init__(self, cfg):
        from ase.calculators.nwchem import NWChem
        self.NWChem = NWChem
        self.profile = NWChem.load_argv_profile(cfg, 'nwchem')

    def version(self):
        stdout = read_stdout(self.profile.argv, createfile='nwchem.nw')
        match = re.search(
            r'Northwest Computational Chemistry Package \(NWChem\) (\S+)',
            stdout,
            re.M,
        )
        return match.group(1)

    def calc(self, **kwargs):
        return self.NWChem(profile=self.profile, **kwargs)

    def socketio_kwargs(self, unixsocket):
        return dict(
            theory='scf',
            task='optimize',
            driver={'socket': {'unix': unixsocket}},
        )


@factory('plumed')
class PlumedFactory:
    def __init__(self, cfg):
        try:
            import plumed
        except ModuleNotFoundError:
            raise NotInstalled

        self.path = plumed.__spec__.origin

    def calc(self, **kwargs):
        from ase.calculators.plumed import Plumed

        return Plumed(**kwargs)


legacy_factory_calculator_names = {
    'ace',
    'amber',
    'crystal',
    'demon',
    'demonnano',
    'dmol',
    'gamess_us',
    'gaussian',
    'gulp',
    'hotbit',
    'onetep',
    'qchem',
    'turbomole',
}


class NoSuchCalculator(Exception):
    pass


class Factories:
    all_calculators = set(calculator_names)
    builtin_calculators = builtin
    autoenabled_calculators = {'asap'} | builtin_calculators

    # Calculators requiring ase-datafiles.
    # TODO: So far hard-coded but should be automatically detected.
    datafile_calculators = {
        'abinit',
        'dftb',
        'elk',
        'espresso',
        'eam',
        'lammpsrun',
        'lammpslib',
        'openmx',
        'siesta',
    }

    def __init__(self, requested_calculators):
        self.machine_info = MachineInformation()
        cfg = self.machine_info.cfg

        factories = {}
        why_not = {}

        from ase.calculators.genericfileio import BadConfiguration
        for name, cls in factory_classes.items():
            try:
                factories[name] = cls(cfg)
            except KeyError as err:
                # XXX FIXME too silent
                why_not[name] = err
            except (NotInstalled, BadConfiguration) as err:
                # FIXME: we should have exactly one kind of error
                # for 'not installed'
                why_not[name] = err
            else:
                why_not[name] = None

        self.factories = factories
        self.why_not = why_not

        requested_calculators = set(requested_calculators)
        if 'auto' in requested_calculators:
            requested_calculators.remove('auto')
            # auto can only work with calculators whose configuration
            # we actually control, so no legacy factories
            requested_calculators |= (
                set(self.factories) - legacy_factory_calculator_names)

        self.requested_calculators = requested_calculators

        for name in self.requested_calculators:
            if name not in self.all_calculators:
                raise NoSuchCalculator(name)

    def installed(self, name):
        return name in self.builtin_calculators | set(self.factories)

    def why_not_installed(self, name):
        return self.why_not[name]

    def is_adhoc(self, name):
        return name not in factory_classes

    def optional(self, name):
        return name not in self.builtin_calculators

    def enabled(self, name):
        auto = name in self.autoenabled_calculators and self.installed(name)
        return auto or (name in self.requested_calculators)

    def require(self, name):
        # XXX This is for old-style calculator tests.
        # Newer calculator tests would depend on a fixture which would
        # make them skip.
        # Older tests call require(name) explicitly.
        assert name in calculator_names
        if not self.installed(name) and not self.is_adhoc(name):
            pytest.skip(f'Not installed: {name}')
        if name not in self.requested_calculators:
            pytest.skip(f'Use --calculators={name} to enable')

    def __getitem__(self, name):
        return self.factories[name]


def get_factories(pytestconfig):
    opt = pytestconfig.getoption('--calculators')
    requested_calculators = opt.split(',') if opt else []
    return Factories(requested_calculators)


def parametrize_calculator_tests(metafunc):
    """Parametrize tests using our custom markers.

    We want tests marked with @pytest.mark.calculator(names) to be
    parametrized over the named calculator or calculators."""
    calculator_inputs = []

    for marker in metafunc.definition.iter_markers(name='calculator'):
        calculator_names = marker.args
        kwargs = dict(marker.kwargs)
        marks = kwargs.pop('marks', [])
        for name in calculator_names:
            param = pytest.param((name, kwargs), marks=marks)
            calculator_inputs.append(param)

    if calculator_inputs:
        metafunc.parametrize(
            'factory',
            calculator_inputs,
            indirect=True,
            ids=lambda input: input[0],
        )


class CalculatorInputs:
    def __init__(self, factory, parameters=None):
        if parameters is None:
            parameters = {}
        self.parameters = parameters
        self.factory = factory

    @property
    def name(self):
        return self.factory.name

    def __repr__(self):
        cls = type(self)
        return f'{cls.__name__}({self.name}, {self.parameters})'

    def new(self, **kwargs):
        kw = dict(self.parameters)
        kw.update(kwargs)
        return CalculatorInputs(self.factory, kw)

    def socketio(self, unixsocket, **kwargs):
        if hasattr(self.factory, 'socketio'):
            kwargs = {**self.parameters, **kwargs}
            return self.factory.socketio(unixsocket, **kwargs)
        from ase.calculators.socketio import SocketIOCalculator

        kwargs = {
            **self.factory.socketio_kwargs(unixsocket),
            **self.parameters,
            **kwargs,
        }
        calc = self.factory.calc(**kwargs)
        return SocketIOCalculator(calc, unixsocket=unixsocket)

    def calc(self, **kwargs):
        param = dict(self.parameters)
        param.update(kwargs)
        return self.factory.calc(**param)
