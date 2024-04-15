# Note:
# Try to avoid module level import statements here to reduce
# import time during CLI execution


class CLICommand:
    """Print information about files or system.

    Without any filename(s), informations about the ASE installation will be
    shown (Python version, library versions, ...).

    With filename(s), the file format will be determined for each file.
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('filename', nargs='*',
                            help='Name of file to determine format for.')
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='Show more information about files.')
        parser.add_argument('--formats', action='store_true',
                            help='List file formats known to ASE.')
        parser.add_argument('--config', action='store_true',
                            help='List configured calculators')
        parser.add_argument('--calculators', action='store_true',
                            help='List all calculators known to ASE '
                            'and whether/how each is installed.  Also, '
                            'attempt to determine version numbers by '
                            'running binaries or importing packages as '
                            'appropriate.')

    @staticmethod
    def run(args):
        from ase.config import cfg
        from ase.io.bundletrajectory import print_bundletrajectory_info
        from ase.io.formats import UnknownFileTypeError, filetype, ioformats
        from ase.io.ulm import print_ulm_info
        if not args.filename:
            print_info()
            if args.formats:
                print()
                print_formats()
            if args.config:
                print()
                cfg.print_everything()
            if args.calculators:
                print()
                cfg.check_calculators()
                # print()
                # from ase.calculators.autodetect import (detect_calculators,
                #                                        format_configs)
                # configs = detect_calculators()
                # print('Calculators:')
                # for message in format_configs(configs):
                #     print('  {}'.format(message))
                # print()
                # print('Available: {}'.format(','.join(sorted(configs))))
            return

        n = max(len(filename) for filename in args.filename) + 2
        nfiles_not_found = 0
        for filename in args.filename:
            try:
                format = filetype(filename)
            except FileNotFoundError:
                format = '?'
                description = 'No such file'
                nfiles_not_found += 1
            except UnknownFileTypeError:
                format = '?'
                description = '?'
            else:
                if format in ioformats:
                    description = ioformats[format].description
                else:
                    description = '?'

            print('{:{}}{} ({})'.format(filename + ':', n,
                                        description, format))
            if args.verbose:
                if format == 'traj':
                    print_ulm_info(filename)
                elif format == 'bundletrajectory':
                    print_bundletrajectory_info(filename)

        raise SystemExit(nfiles_not_found)


def print_info():
    import platform
    import sys

    from ase.dependencies import all_dependencies

    versions = [('platform', platform.platform()),
                ('python-' + sys.version.split()[0], sys.executable)]

    for name, path in versions + all_dependencies():
        print(f'{name:24} {path}')


def print_formats():
    from ase.io.formats import ioformats

    print('Supported formats:')
    for fmtname in sorted(ioformats):
        fmt = ioformats[fmtname]

        infos = [fmt.modes, 'single' if fmt.single else 'multi']
        if fmt.isbinary:
            infos.append('binary')
        if fmt.encoding is not None:
            infos.append(fmt.encoding)
        infostring = '/'.join(infos)

        moreinfo = [infostring]
        if fmt.extensions:
            moreinfo.append('ext={}'.format('|'.join(fmt.extensions)))
        if fmt.globs:
            moreinfo.append('glob={}'.format('|'.join(fmt.globs)))

        print('  {} [{}]: {}'.format(fmt.name,
                                     ', '.join(moreinfo),
                                     fmt.description))
