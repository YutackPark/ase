from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Mapping

from ase.calculators.abc import GetOutputsMixin
from ase.calculators.calculator import BaseCalculator, EnvironmentError


    def __init__(self, parallel=True, parallel_info=None):
        """
        """
        self.parallel_info = parallel_info
        self.parallel = parallel

    def get_command(self, inputfile) -> Iterable[str]:
        """
        Get the command to run. This should be a list of strings.

        This is main method that needs to be implemented by subclasses.
        """
        command = []
        if self.parallel:
            command.append(self.parallel_info['binary'])

            for key, value in self.parallel_info.items():
                if key == 'binary':
                    continue
                if type(value) is not bool:
                    command.append(f'{key}')
                    command.append(f'{value}')
                elif value:
                    command.append(f'{key}')

        command.extend(self.get_calculator_command(inputfile))
        return command

    @abstractmethod
    def get_calculator_command(self, inputfile):
        ...

    def run(self, directory, inputfile, outputfile):
        """
        Run the command in the given directory.

        Parameters
        ----------
        directory : pathlib.Path
            The directory to run the command in.
        inputfile : str
            The name of the input file.
        outputfile : str
            The name of the output file.
        """

        from subprocess import check_call
        import os

        argv_command = self.get_command(inputfile)
        with open(directory / outputfile, "wb") as fd:
            check_call(argv_command, cwd=directory, stdout=fd, env=os.environ)

    @abstractmethod
    def version(self):
        """
        Get the version of the code.

        Returns
        -------
        str
            The version of the code.
        """
        ...

    @classmethod
    def from_config(cls, cfg, section_name):
        """
        Create a profile from a configuration file.

        Parameters
        ----------
        cfg : ase.config.Config
            The configuration object.
        section_name : str
            The name of the section in the configuration file. E.g. the name
            of the template that this profile is for.

        Returns
        -------
        BaseProfile
            The profile object.
        """
        return cls(**cfg.parser["general"], **cfg.parser[section_name])


def read_stdout(args, createfile=None):
    """Run command in tempdir and return standard output.

    Helper function for getting version numbers of DFT codes.
    Most DFT codes don't implement a --version flag, so in order to
    determine the code version, we just run the code until it prints
    a version number."""
    import tempfile
    from subprocess import PIPE, Popen

    with tempfile.TemporaryDirectory() as directory:
        if createfile is not None:
            path = Path(directory) / createfile
            path.touch()
        proc = Popen(
            args,
            stdout=PIPE,
            stderr=PIPE,
            stdin=PIPE,
            cwd=directory,
            encoding="ascii",
        )
        stdout, _ = proc.communicate()
        # Exit code will be != 0 because there isn't an input file
    return stdout


class CalculatorTemplate(ABC):
    def __init__(self, name: str, implemented_properties: Iterable[str]):
        self.name = name
        self.implemented_properties = frozenset(implemented_properties)

    @abstractmethod
    def write_input(self, directory, atoms, parameters, properties):
        ...

    @abstractmethod
    def execute(self, directory, profile):
        ...

    @abstractmethod
    def read_results(self, directory: PathLike) -> Mapping[str, Any]:
        ...

    @abstractmethod
    def load_profile(self, cfg, parallel_info={}, parallel=True):
        ...

    def socketio_calculator(
        self,
        profile,
        parameters,
        directory,
        # We may need quite a few socket kwargs here
        # if we want to expose all the timeout etc. from
        # SocketIOCalculator.
        unixsocket=None,
        port=None,
    ):
        import os
        from subprocess import Popen

        from ase.calculators.socketio import SocketIOCalculator

        if port and unixsocket:
            raise TypeError(
                "For the socketio_calculator only a UNIX "
                "(unixsocket) or INET (port) socket can be used"
                " not both."
            )

        if not port and not unixsocket:
            raise TypeError(
                "For the socketio_calculator either a "
                "UNIX (unixsocket) or INET (port) socket "
                "must be used"
            )

        if not (
            hasattr(self, "socketio_argv")
            and hasattr(self, "socketio_parameters")
        ):
            raise TypeError(
                f"Template {self} does not implement mandatory "
                "socketio_argv() and socketio_parameters()"
            )

        # XXX need socketio ABC or something
        argv = self.socketio_argv(profile, unixsocket, port)
        parameters = {
            **self.socketio_parameters(unixsocket, port),
            **parameters,
        }

        # Not so elegant that socket args are passed to this function
        # via socketiocalculator when we could make a closure right here.
        def launch(atoms, properties, port, unixsocket):
            directory.mkdir(exist_ok=True, parents=True)

            self.write_input(
                atoms=atoms,
                parameters=parameters,
                properties=properties,
                directory=directory,
            )

            with open(directory / self.outputname, "w") as out_fd:
                return Popen(argv, stdout=out_fd, cwd=directory, env=os.environ)

        return SocketIOCalculator(
            launch_client=launch, unixsocket=unixsocket, port=port
        )


class GenericFileIOCalculator(BaseCalculator, GetOutputsMixin):
    def __init__(self, *, template, profile, directory, parameters=None, parallel_info=None, 
                parallel=True):
        self.template = template

        if profile is None:
            from ase.config import cfg

            parallel_config = dict(cfg.parser['parallel'])
            variable_whitelist = ['binary']
            for key in parallel_config:
                assert key in variable_whitelist, f"Unknown variable {key} in section [parallel]"

            parallel_info = parallel_info if parallel_info is not None else {}
            parallel_config.update(parallel_info)

            if template.name not in cfg.parser:
                raise EnvironmentError(f"No configuration of {template.name}")
            myconfig = cfg.parser[template.name]
            try:
                profile = template.load_profile(myconfig, parallel_config, parallel=parallel)
            except Exception as err:
                # configvars = dict(myconfig)
                raise EnvironmentError(
                    f"Failed to load section [{template.name}] "
                    "from configuration: {configvars}"
                ) from err

        self.profile = profile

        # Maybe we should allow directory to be a factory, so
        # calculators e.g. produce new directories on demand.
        self.directory = Path(directory)

        super().__init__(parameters)

    def set(self, *args, **kwargs):
        raise RuntimeError(
            "No setting parameters for now, please.  "
            "Just create new calculators."
        )

    def __repr__(self):
        return "{}({})".format(type(self).__name__, self.template.name)

    @property
    def implemented_properties(self):
        return self.template.implemented_properties

    @property
    def name(self):
        return self.template.name

    def write_inputfiles(self, atoms, properties):
        # SocketIOCalculators like to write inputfiles
        # without calculating.
        self.directory.mkdir(exist_ok=True, parents=True)
        self.template.write_input(
            profile=self.profile,
            atoms=atoms,
            parameters=self.parameters,
            properties=properties,
            directory=self.directory,
        )

    def calculate(self, atoms, properties, system_changes):
        self.write_inputfiles(atoms, properties)
        self.template.execute(self.directory, self.profile)
        self.results = self.template.read_results(self.directory)
        # XXX Return something useful?

    def _outputmixin_get_results(self):
        return self.results

    def socketio(self, **socketkwargs):
        return self.template.socketio_calculator(
            directory=self.directory,
            parameters=self.parameters,
            profile=self.profile,
            **socketkwargs,
        )
