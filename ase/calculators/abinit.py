"""This module defines an ASE interface to ABINIT.

http://www.abinit.org/
"""

import re

import os
import ase.io.abinit as io
from ase.calculators.genericfileio import (CalculatorTemplate,
                                           GenericFileIOCalculator)
from subprocess import check_output, check_call
from pathlib import Path


def get_abinit_version(command):
    txt = check_output([command, '--version']).decode('ascii')
    # This allows trailing stuff like betas, rc and so
    m = re.match(r'\s*(\d\.\d\.\d)', txt)
    if m is None:
        raise RuntimeError('Cannot recognize abinit version. '
                           'Start of output: {}'
                           .format(txt[:40]))
    return m.group(1)


class AbinitProfile:
    def __init__(self, argv):
        self.argv = argv

    def version(self):
        return check_output(self.argv + ['--version'])

    def run(self, directory, inputfile, outputfile):
        argv = self.argv + [str(inputfile)]
        with open(directory / outputfile, 'wb') as fd:
            check_call(argv, stdout=fd, env=os.environ, cwd=directory)

    def socketio_argv_unix(self, socket):
        # XXX clean up the passing of the inputfile
        inputfile = AbinitTemplate().input_file
        return [*self.argv, inputfile, '--ipi', f'{socket}:UNIX']


class AbinitTemplate(CalculatorTemplate):
    _label = 'abinit'  # Controls naming of files within calculation directory

    def __init__(self):
        super().__init__(
            name='abinit',
            implemented_properties=['energy', 'free_energy',
                                    'forces', 'stress', 'magmom'])

        self.input_file = f'{self._label}.in'
        self.output_file = f'{self._label}.log'

    def execute(self, directory, profile) -> None:
        profile.run(directory, self.input_file, self.output_file)

    def write_input(self, directory, atoms, parameters, properties):
        directory = Path(directory)
        parameters = dict(parameters)
        pp_paths = parameters.pop('pp_paths', None)
        assert pp_paths is not None

        kw = dict(
            xc='LDA',
            smearing=None,
            kpts=None,
            raw=None,
            pps='fhi')
        kw.update(parameters)

        io.prepare_abinit_input(
            directory=directory,
            atoms=atoms, properties=properties, parameters=kw,
            pp_paths=pp_paths)

    def read_results(self, directory):
        return io.read_abinit_outputs(directory, self._label)


class Abinit(GenericFileIOCalculator):
    """Class for doing ABINIT calculations.

    The default parameters are very close to those that the ABINIT
    Fortran code would use.  These are the exceptions::

      calc = Abinit(label='abinit', xc='LDA', ecut=400, toldfe=1e-5)
    """

    def __init__(self, *, profile=None, directory='.', **kwargs):
        """Construct ABINIT-calculator object.

        Parameters
        ==========
        label: str
            Prefix to use for filenames (label.in, label.txt, ...).
            Default is 'abinit'.

        Examples
        ========
        Use default values:

        >>> h = Atoms('H', calculator=Abinit(ecut=200, toldfe=0.001))
        >>> h.center(vacuum=3.0)
        >>> e = h.get_potential_energy()

        """

        if profile is None:
            profile = AbinitProfile(['abinit'])

        super().__init__(template=AbinitTemplate(),
                         profile=profile,
                         directory=directory,
                         parameters=kwargs)
