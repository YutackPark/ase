from os import PathLike
from pathlib import Path
from typing import Iterable, Mapping, Any
from abc import ABC, abstractmethod

from ase.io import read, write
from ase.calculators.abc import GetOutputsMixin
from ase.calculators.calculator import BaseCalculator


def read_stdout(args, createfile=None):
    """Run command in tempdir and return standard output.

    Helper function for getting version numbers of DFT codes.
    Most DFT codes don't implement a --version flag, so in order to
    determine the code version, we just run the code until it prints
    a version number."""
    import tempfile
    from subprocess import Popen, PIPE
    with tempfile.TemporaryDirectory() as directory:
        if createfile is not None:
            path = Path(directory) / createfile
            path.touch()
        proc = Popen(args,
                     stdout=PIPE,
                     stderr=PIPE,
                     stdin=PIPE,
                     cwd=directory,
                     encoding='ascii')
        stdout, _ = proc.communicate()
        # Exit code will be != 0 because there isn't an input file
    return stdout


class CalculatorTemplate(ABC):
    def __init__(self, name: str, implemented_properties: Iterable[str]):
        self.name = name
        self.implemented_properties = set(implemented_properties)

    @abstractmethod
    def write_input(self, directory, atoms, parameters, properties):
        ...

    def execute(self, profile, directory: PathLike) -> None:
        # Should be abstract?
        profile.run(directory,
                    self.input_file,
                    self.output_file)

    @abstractmethod
    def read_results(self, directory: PathLike) -> Mapping[str, Any]:
        ...


class EspressoTemplate(CalculatorTemplate):
    def __init__(self):
        super().__init__('espresso', ['energy', 'forces', 'stress', 'magmoms'])
        self.inputname = 'espresso.pwi'
        self.outputname = 'espresso.pwo'

    def write_input(self, directory, atoms, parameters, properties):
        directory.mkdir(exist_ok=True, parents=True)
        dst = directory / self.inputname
        write(dst, atoms, format='espresso-in', properties=properties,
              **parameters)

    def execute(self, profile, directory):
        profile.run(directory,
                    self.inputname,
                    self.outputname)

    def read_results(self, directory):
        path = directory / self.outputname
        atoms = read(path, format='espresso-out')
        return dict(atoms.calc.properties())


class GenericFileIOCalculator(BaseCalculator, GetOutputsMixin):
    def __init__(self, template, profile, directory='.', parameters=None):
        self.template = template
        self.profile = profile

        # Maybe we should allow directory to be a factory, so
        # calculators e.g. produce new directories on demand.
        self.directory = Path(directory)

        super().__init__(parameters)

    def set(self, *args, **kwargs):
        raise RuntimeError('No setting parameters for now, please.  '
                           'Just create new calculators.')

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, self.template.name)

    def write_input(self, atoms, properties, system_changes):
        # XXX for socketio compatibility; remove later
        self.template.write_input(self.directory, atoms,
                                  self.parameters, properties)

    @property
    def implemented_properties(self):
        return self.template.implemented_properties

    @property
    def name(self):
        return self.template.name

    def calculate(self, atoms, properties, system_changes):
        self.atoms = atoms.copy()

        directory = self.directory

        self.template.write_input(directory, atoms, self.parameters,
                                  properties)
        self.template.execute(self.profile, directory)
        self.results = self.template.read_results(directory)
        # XXX Return something useful?

    def _outputmixin_get_results(self):
        return self.results
