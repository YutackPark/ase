import re
import ase.io.orca as io
from ase.calculators.genericfileio import (CalculatorTemplate,
                                           GenericFileIOCalculator)


def get_version_from_orca_header(orca_header):
    match = re.search(r'Program Version (\S+)', orca_header, re.M)
    return match.group(1)


class OrcaProfile:
    def __init__(self, argv):
        self.argv = argv

    def version(self):
        # XXX Allow MPI in argv; the version call should not be parallel.
        from ase.calculators.genericfileio import read_stdout
        stdout = read_stdout([*self.argv, "does_not_exist"])
        return get_version_from_orca_header(stdout)

    def run(self, directory, inputfile, outputfile):
        from subprocess import check_call
        with open(outputfile, 'w') as fd:
            check_call(self.argv + [str(inputfile)], stdout=fd, cwd=directory)


class OrcaTemplate(CalculatorTemplate):
    _label = 'orca'

    def __init__(self):
        super().__init__(name='orca',
                         implemented_properties=['energy', 'free_energy',
                                                 'forces'])

        self.input_file = f'{self._label}.inp'
        self.output_file = f'{self._label}.out'

    def execute(self, directory, profile) -> None:
        profile.run(directory, self.input_file, self.output_file)

    def write_input(self, directory, atoms, parameters, properties):
        parameters = dict(parameters)

        kw = dict(charge=0, mult=1, orcasimpleinput='B3LYP def2-TZVP',
                  orcablocks='%pal nprocs 1 end')
        kw.update(parameters)

        io.write_orca(directory / self.input_file, atoms, kw)

    def read_results(self, directory):
        return io.read_orca_outputs(directory, directory / self.output_file)


class ORCA(GenericFileIOCalculator):
    """Class for doing Orca calculations.

    Example:

      calc = ORCA(charge=0, mult=1, orcasimpleinput='B3LYP def2-TZVP',
        orcablocks='%pal nprocs 16 end')
    """

    def __init__(self, *, profile=None, directory='.', **kwargs):
        """Construct ORCA-calculator object.

        Parameters
        ==========
        charge: int

        mult: int

        orcasimpleinput : str

        orcablocks: str


        Examples
        ========
        Use default values:


        >>> h = Atoms('H', calculator=ORCA(charge=0,mult=1,directory='water',
        orcasimpleinput='B3LYP def2-TZVP',
        orcablocks='%pal nprocs 16 end')

        """

        if profile is None:
            profile = OrcaProfile(['orca'])

        super().__init__(template=OrcaTemplate(),
                         profile=profile, directory=directory,
                         parameters=kwargs)
