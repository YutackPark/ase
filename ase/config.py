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

        parser = configparser.ConfigParser(converters={"argv": argv_converter})
        envpath = os.environ.get("ASE_CONFIG_PATH")
        if envpath is not None:
            paths = [Path(p) for p in envpath.split(":")]
        else:
            paths = [Path.home() / ".config/ase/config.ini"]
        loaded_paths = parser.read(paths)
        return loaded_paths, parser

    @property
    def paths(self):
        return self.paths_and_parser()[0]

    @property
    def parser(self):
        return self.paths_and_parser()[1]

    def check_calculators(self):
        from ase.calculators.names import names, templates

        print("Calculators")
        print("===========")
        print()
        print("Configured in ASE")
        print("   |  Installed on machine")
        print("   |   |  Name & version")
        print("   |   |  |")
        for name in names:
            # configured = False
            # installed = False
            template = templates.get(name)
            # if template is None:
            # XXX no template for this calculator.
            # We need templates for all calculators somehow,
            # but we can probably generate those for old FileIOCalculators
            # automatically.
            #    continue

            fullname = name
            try:
                codeconfig = self.parser[name]
            except KeyError:
                codeconfig = None
                version = None
            else:
                if template is None:
                    codeconfig = None  # XXX we should not be executing this
                    version = None
                else:
                    profile = template.load_profile(codeconfig)
                    # XXX should be made robust to failure here:
                    version = profile.version()
                    fullname = f"{name}-{version}"

            def tickmark(thing):
                return "[ ]" if thing is None else "[x]"

            msg = "  {configured} {installed} {fullname}".format(
                configured=tickmark(codeconfig),
                installed=tickmark(version),
                fullname=fullname,
            )
            print(msg)

    def print_everything(self):
        print("Configuration")
        print("-------------")
        print()
        if not cfg.paths:
            print("No configuration loaded.")

        for path in cfg.paths:
            print(f"Loaded: {path}")

        print()
        for name, section in cfg.parser.items():
            print(name)
            if not section:
                print("  (Nothing configured)")
            for key, val in section.items():
                print(f"  {key}: {val}")
            print()


cfg = Config()
