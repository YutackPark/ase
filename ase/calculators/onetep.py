"""ONETEP interface for the Atomic Simulation Environment (ASE) package

T. Demeyere, T.Demeyere@soton.ac.uk (2023)

https://onetep.org"""

from pathlib import Path

from ase.calculators.genericfileio import (CalculatorTemplate,
                                           GenericFileIOCalculator,
                                           read_stdout,
                                           BaseProfile)
from ase.io import read, write

# TARP: New setup exc should be the onetep binary itself
# def find_onetep_command(cmd):
#     exploded_cmd = cmd.split()
#     n_cmd = len(exploded_cmd)
#     if n_cmd == 1:
#         return cmd
#     for substring in exploded_cmd:
#         if 'onetep' in substring:
#             return substring
#     return None


class OnetepProfile(BaseProfile):
    def __init__(self, exc, **kwargs):
        super().__init__(**kwargs)
        self.exc = exc

    def version(self):
        # onetep_exec = find_onetep_command(self.argv)
        lines = read_stdout(self.exc)
        return self.parse_version(lines)

    def parse_version(lines):
        return '1.0.0'

    def get_calculator_command(self, inputfile):
        return [self.exc, str(inputfile)]


class OnetepTemplate(CalculatorTemplate):
    def __init__(self, label, append):
        super().__init__(
            name='ONETEP',
            implemented_properties=[
                'energy',
                'free_energy',
                'forces',
                'stress'])
        self.label = label
        self.input = label + '.dat'
        self.output = label + '.out'
        self.error = label + '.err'
        self.append = append

    def execute(self, directory, profile):
        profile.run(directory, self.input, self.output, self.error,
                    self.append)

    def read_results(self, directory):
        output_path = directory / self.output
        atoms = read(output_path, format='onetep-out')
        return dict(atoms.calc.properties())

    def write_input(self, directory, atoms, parameters, properties):
        input_path = directory / self.input
        write(input_path, atoms, format='onetep-in',
              properties=properties, **parameters)

    def load_profile(self, cfg, **kwargs):
        return OnetepProfile.from_config(cfg, self.name, **kwargs)


class Onetep(GenericFileIOCalculator):
    """
    Class for the ONETEP calculator, uses ase/io/onetep.py.
    Need the env variable "ASE_ONETEP_COMMAND" defined to
    properly work. All other options are passed in kwargs.

    Parameters
    ----------
    autorestart : Bool
        When activated, manages restart keywords automatically.
    append: Bool
        Append to output instead of overwriting.
    directory: str
        Directory where to run the calculation(s).
    keywords: dict
        Dictionary with ONETEP keywords to write,
        keywords with lists as values will be
        treated like blocks, with each element
        of list being a different line.
    label: str
        Name used for the ONETEP prefix.
    xc: str
        DFT xc to use e.g (PBE, RPBE, ...).
    ngwfs_count: int|list|dict
        Behaviour depends on the type:
            int: every species will have this amount
            of ngwfs.
            list: list of int, will be attributed
            alphabetically to species:
            dict: keys are species name(s),
            value are their number:
    ngwfs_radius: int|list|dict
        Behaviour depends on the type:
            float: every species will have this radius.
            list: list of float, will be attributed
            alphabetically to species:
            [10.0, 9.0]
            dict: keys are species name(s),
            value are their radius:
            {'Na': 9.0, 'Cl': 10.0}
    pseudopotentials: list|dict
        Behaviour depends on the type:
            list: list of string(s), will be attributed
            alphabetically to specie(s):
            ['Cl.usp', 'Na.usp']
            dict: keys are species name(s) their
            value are the pseudopotential file to use:
            {'Na': 'Na.usp', 'Cl': 'Cl.usp'}
    pseudo_path: str
        Where to look for pseudopotential, correspond
        to the pseudo_path keyword of ONETEP.

        .. note::
           write_forces is always turned on by default
           when using this interface.

        .. note::
           Little to no check is performed on the keywords provided by the user
           via the keyword dictionary, it is the user responsibility that they
           are valid ONETEP keywords.
    """
    # TARP: I thought GenericFileIO calculators no longer had  atoms attached
    # to them
    def __init__(
            self,
            label='onetep',
            directory='.',
            profile=None,
            append=False,
            autorestart=True,
            atoms=None,
            parallel_info=None,
            parallel=True,
            **kwargs):

        self.directory = Path(directory)
        self.autorestart = autorestart
        self.label = label
        self.keywords = kwargs.get('keywords', None)
        self.append = append
        self.template = OnetepTemplate(label, append=self.append)

        kwargs['autorestart'] = self.autorestart
        kwargs['directory'] = self.directory
        kwargs['label'] = self.label
        super().__init__(profile=profile, template=self.template,
                         directory=directory,
                         parameters=kwargs,
                         parallel=parallel,
                         parallel_info=parallel_info)
        # Copy is probably not needed, but just in case
        if atoms is not None:
            self.atoms = atoms.copy()
            if atoms.calc is not None:
                # TARP: Does this make sense? Why append the previous
                # calculators results to a new calculator. Especially
                # without checking if the parameters are the same as well.
                self.results = atoms.calc.results.copy()
