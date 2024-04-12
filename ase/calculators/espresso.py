"""Quantum ESPRESSO Calculator

Run pw.x jobs.
"""


import os
import warnings
from pathlib import Path

from ase.calculators.genericfileio import (BaseProfile, CalculatorTemplate,
                                           GenericFileIOCalculator,
                                           read_stdout)
from ase.io import read, write
from ase.io.espresso import Namelist

compatibility_msg = (
    'Espresso calculator is being restructured.  Please use e.g. '
    "Espresso(profile=EspressoProfile(argv=['mpiexec', 'pw.x'])) "
    'to customize command-line arguments.'
)


# XXX We should find a way to display this warning.
# warn_template = 'Property "%s" is None. Typically, this is because the ' \
#                 'required information has not been printed by Quantum ' \
#                 'Espresso at a "low" verbosity level (the default). ' \
#                 'Please try running Quantum Espresso with "high" verbosity.'


class EspressoProfile(BaseProfile):
    def __init__(self, binary, pseudo_dir, **kwargs):
        super().__init__(**kwargs)
        self.binary = binary
        self.pseudo_dir = Path(pseudo_dir)

    @staticmethod
    def parse_version(stdout):
        import re

        match = re.match(r'\s*Program PWSCF\s*v\.(\S+)', stdout, re.M)
        assert match is not None
        return match.group(1)

    def version(self):
        try:
            stdout = read_stdout(self.binary)
            return self.parse_version(stdout)
        except FileNotFoundError:
            warnings.warn(
                f'The executable {self.binary} is not found on the path'
            )
            return None

    def get_calculator_command(self, inputfile):
        return [self.binary, '-in', inputfile]


class EspressoTemplate(CalculatorTemplate):
    _label = 'espresso'

    def __init__(self):
        super().__init__(
            'espresso',
            ['energy', 'free_energy', 'forces', 'stress', 'magmoms', 'dipole'],
        )
        self.inputname = f'{self._label}.pwi'
        self.outputname = f'{self._label}.pwo'
        self.errorname = f"{self._label}.err"

    def write_input(self, profile, directory, atoms, parameters, properties):
        dst = directory / self.inputname

        input_data = Namelist(parameters.pop("input_data", None))
        input_data.to_nested("pw")
        input_data["control"].setdefault("pseudo_dir", str(profile.pseudo_dir))

        parameters["input_data"] = input_data

        write(
            dst,
            atoms,
            format='espresso-in',
            properties=properties,
            **parameters,
        )

    def execute(self, directory, profile):
        profile.run(directory, self.inputname, self.outputname,
                    errorfile=self.errorname)

    def read_results(self, directory):
        path = directory / self.outputname
        atoms = read(path, format='espresso-out')
        return dict(atoms.calc.properties())

    def load_profile(self, cfg, **kwargs):
        return EspressoProfile.from_config(cfg, self.name, **kwargs)

    def socketio_parameters(self, unixsocket, port):
        return {}

    def socketio_argv(self, profile, unixsocket, port):
        if unixsocket:
            ipi_arg = f'{unixsocket}:UNIX'
        else:
            ipi_arg = f'localhost:{port:d}'  # XXX should take host, too
        return profile.get_calculator_command(self.inputname) + [
            '--ipi',
            ipi_arg,
        ]


class Espresso(GenericFileIOCalculator):
    def __init__(
        self,
        *,
        profile=None,
        command=GenericFileIOCalculator._deprecated,
        label=GenericFileIOCalculator._deprecated,
        directory='.',
        parallel_info=None,
        parallel=True,
        **kwargs,
    ):
        """
        All options for pw.x are copied verbatim to the input file, and put
        into the correct section. Use ``input_data`` for parameters that are
        already in a dict, all other ``kwargs`` are passed as parameters.

        Accepts all the options for pw.x as given in the QE docs, plus some
        additional options:

        input_data: dict
            A flat or nested dictionary with input parameters for pw.x
        pseudopotentials: dict
            A filename for each atomic species, e.g.
            ``{'O': 'O.pbe-rrkjus.UPF', 'H': 'H.pbe-rrkjus.UPF'}``.
            A dummy name will be used if none are given.
        kspacing: float
            Generate a grid of k-points with this as the minimum distance,
            in A^-1 between them in reciprocal space. If set to None, kpts
            will be used instead.
        kpts: (int, int, int), dict, or BandPath
            If kpts is a tuple (or list) of 3 integers, it is interpreted
            as the dimensions of a Monkhorst-Pack grid.
            If ``kpts`` is set to ``None``, only the Γ-point will be included
            and QE will use routines optimized for Γ-point-only calculations.
            Compared to Γ-point-only calculations without this optimization
            (i.e. with ``kpts=(1, 1, 1)``), the memory and CPU requirements
            are typically reduced by half.
            If kpts is a dict, it will either be interpreted as a path
            in the Brillouin zone (*) if it contains the 'path' keyword,
            otherwise it is converted to a Monkhorst-Pack grid (**).
            (*) see ase.dft.kpoints.bandpath
            (**) see ase.calculators.calculator.kpts2sizeandoffsets
        koffset: (int, int, int)
            Offset of kpoints in each direction. Must be 0 (no offset) or
            1 (half grid offset). Setting to True is equivalent to (1, 1, 1).


        .. note::
           Set ``tprnfor=True`` and ``tstress=True`` to calculate forces and
           stresses.

        .. note::
           Band structure plots can be made as follows:


           1. Perform a regular self-consistent calculation,
              saving the wave functions at the end, as well as
              getting the Fermi energy:

              >>> input_data = {<your input data>}
              >>> calc = Espresso(input_data=input_data, ...)
              >>> atoms.calc = calc
              >>> atoms.get_potential_energy()
              >>> fermi_level = calc.get_fermi_level()

           2. Perform a non-self-consistent 'band structure' run
              after updating your input_data and kpts keywords:

              >>> input_data['control'].update({'calculation':'bands',
              >>>                               'restart_mode':'restart',
              >>>                               'verbosity':'high'})
              >>> calc.set(kpts={<your Brillouin zone path>},
              >>>          input_data=input_data)
              >>> calc.calculate(atoms)

           3. Make the plot using the BandStructure functionality,
              after setting the Fermi level to that of the prior
              self-consistent calculation:

              >>> bs = calc.band_structure()
              >>> bs.reference = fermi_energy
              >>> bs.plot()

        """

        if command is not self._deprecated:
            raise RuntimeError(compatibility_msg)

        if label is not self._deprecated:
            import warnings

            warnings.warn(
                'Ignoring label, please use directory instead', FutureWarning
            )

        if 'ASE_ESPRESSO_COMMAND' in os.environ and profile is None:
            import warnings

            warnings.warn(compatibility_msg, FutureWarning)

        template = EspressoTemplate()
        super().__init__(
            profile=profile,
            template=template,
            directory=directory,
            parallel_info=parallel_info,
            parallel=parallel,
            parameters=kwargs,
        )
