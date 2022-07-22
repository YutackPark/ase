import os
import configparser
from pathlib import Path

from ase.utils import lazymethod
import shlex


class Config:
    @lazymethod
    def paths_and_parser(self):
        def argv_converter(argv):
            return shlex.split(argv)

        parser = configparser.ConfigParser(
            converters={'argv': argv_converter})
        envpath = os.environ.get('ASE_CONFIG_PATH')
        if envpath is not None:
            paths = [Path(p) for p in envpath.split(':')]
        else:
            paths = [Path.home() / '.config/ase/config.ini']
        loaded_paths = parser.read(paths)
        return loaded_paths, parser

    @property
    def paths(self):
        return self.paths_and_parser()[0]

    @property
    def parser(self):
        return self.paths_and_parser()[1]

    def print_everything(self):
        print('Configuration')
        print('-------------')
        print()
        if not cfg.paths:
            print('No configuration loaded.')

        for path in cfg.paths:
            print(f'Loaded: {path}')

        print()
        for name, section in cfg.parser.items():
            print(name)
            if not section:
                print('  (Nothing configured)')
            for key, val in section.items():
                print(f'  {key}: {val}')
            print()


cfg = Config()
