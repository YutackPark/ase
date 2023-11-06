import re

import ase.io.orca as io
from ase.calculators.genericfileio import (CalculatorTemplate,
                                           GenericFileIOCalculator,
                                           BaseProfile)


def get_version_from_orca_header(orca_header):
    match = re.search(r'Program Version (\S+)', orca_header, re.M)
    return match.group(1)


class OrcaProfile(BaseProfile):
    def __init__(self, binary, **kwargs):
        """
        Parameters
        ----------
        binary : str
            Full path to the orca binary, if full path is not specified ORCA
            cannot run in parallel.
        """
        # Because ORCA handles its parallelization without being called with
        # mpirun/mpiexec/etc parallel should be set to False.
        # Whether or not it is run in parallel is controlled by the orcablocks
        super().__init__(parallel=False, parallel_info={})
        self.binary = binary

    def version(self):
        # XXX Allow MPI in argv; the version call should not be parallel.
        from ase.calculators.genericfileio import read_stdout
        stdout = read_stdout([self.binary, "does_not_exist"])
        return get_version_from_orca_header(stdout)

    def get_calculator_command(self, inputfile):
        return [self.binary, inputfile]


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

    def write_input(self, profile, directory, atoms, parameters, properties):
        parameters = dict(parameters)

        kw = dict(charge=0, mult=1, orcasimpleinput='B3LYP def2-TZVP',
                  orcablocks='%pal nprocs 1 end')
        kw.update(parameters)

        io.write_orca(directory / self.input_file, atoms, kw)

    def read_results(self, directory):
        return io.read_orca_outputs(directory, directory / self.output_file)

    def load_profile(self, cfg, **kwargs):
        return OrcaProfile.from_config(cfg, self.name, **kwargs)


class ORCA(GenericFileIOCalculator):
    """Class for doing ORCA calculations.

    Example:

      calc = ORCA(charge=0, mult=1, orcasimpleinput='B3LYP def2-TZVP',
        orcablocks='%pal nprocs 16 end')
    """

    def __init__(self, *, profile=None, directory='.', parallel_info=None,
                 parallel=None, **kwargs):
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

        >>> from ase.calculators.orca import ORCA
        >>> h = Atoms(
        ...     'H',
        ...     calculator=ORCA(
        ...         charge=0,
        ...         mult=1,
        ...         directory='water',
        ...         orcasimpleinput='B3LYP def2-TZVP',
        ...         orcablocks='%pal nprocs 16 end'))

        """

        assert parallel is None, \
            'ORCA does not support keyword parallel - use orcablocks'
        assert parallel_info is None, \
            'ORCA does not support keyword parallel_info - use orcablocks'

        super().__init__(template=OrcaTemplate(),
                         profile=profile, directory=directory,
                         parameters=kwargs)
