import os
import sys
from abc import ABC, abstractmethod
import pickle
from subprocess import Popen, PIPE
from ase.calculators.calculator import Calculator, all_properties


class PackedCalculator(ABC):
    """Portable calculator for use via PythonSubProcessCalculator.

    This class allows creating and talking to a calculator which
    exists inside a different process, possibly with MPI or srun.

    Use this when you want to use ASE mostly in serial, but run some
    calculations in a parallel Python environment.

    Most existing calculators can be used this way through the
    NamedPackedCalculator implementation.  To customize the behaviour
    for other calculators, write a custom class inheriting this one.

    Example::

      from ase.build import bulk

      atoms = bulk('Au')
      pack = NamedPackedCalculator('emt')

      with pack.calculator() as atoms.calc:
          energy = atoms.get_potential_energy()

    The computation takes place inside a subprocess which lives as long
    as the with statement.
    """

    @abstractmethod
    def unpack_calculator(self) -> Calculator:
        """Return the calculator packed inside.

        This method will be called inside the subprocess doing
        computations."""

    def calculator(self, mpi_command=None) -> 'PythonSubProcessCalculator':
        """Return a PythonSubProcessCalculator for this calculator.

        The subprocess calculator wraps a subprocess containing
        the actual calculator, and computations are done inside that
        subprocess."""
        return PythonSubProcessCalculator(self, mpi_command=mpi_command)


class NamedPackedCalculator(PackedCalculator):
    """PackedCalculator implementation which works with standard calculators.

    This works with calculators known by ase.calculators.calculator."""

    def __init__(self, name, kwargs=None):
        self._name = name
        if kwargs is None:
            kwargs = {}
        self._kwargs = kwargs

    def unpack_calculator(self):
        from ase.calculators.calculator import get_calculator_class
        cls = get_calculator_class(self._name)
        return cls(**self._kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._name}, {self._kwargs})'


class MPICommand:
    def __init__(self, argv):
        self.argv = argv

    @classmethod
    def python_argv(cls):
        return [sys.executable, '-m', 'ase.calculators.subprocesscalculator']

    @classmethod
    def parallel(cls, nprocs, mpi_argv=tuple()):
        return cls(['mpiexec', '-n', str(nprocs)]
                   + list(mpi_argv)
                   + cls.python_argv()
                   + ['mpi4py'])

    @classmethod
    def serial(cls):
        return MPICommand(cls.python_argv() + ['standard'])

    def execute(self):
        # On this computer (Ubuntu 20.04 + OpenMPI) the subprocess crashes
        # without output during startup if os.environ is not passed along.
        # Hence we pass os.environ.  Not sure if this is a machine thing
        # or in general.  --askhl
        return Popen(self.argv, stdout=PIPE,
                     stdin=PIPE, env=os.environ)


def gpaw_process(ncores=1, **kwargs):
    packed = NamedPackedCalculator('gpaw', kwargs)
    mpicommand = MPICommand([
        sys.executable, '-m', 'gpaw', '-P', str(ncores), 'python', '-m',
        'ase.calculators.subprocesscalculator', 'standard',
    ])
    return PythonSubProcessCalculator(packed, mpicommand)


class PythonSubProcessCalculator(Calculator):
    """Calculator for running calculations in external processes.

    TODO: This should work with arbitrary commands including MPI stuff.

    This calculator runs a subprocess wherein it sets up an
    actual calculator.  Calculations are forwarded through pickle
    to that calculator, which returns results through pickle."""
    implemented_properties = list(all_properties)

    def __init__(self, calc_input, mpi_command=None):
        super().__init__()

        self.proc = None
        self.calc_input = calc_input
        if mpi_command is None:
            mpi_command = MPICommand.serial()
        self.mpi_command = mpi_command

    def set(self, **kwargs):
        if hasattr(self, 'proc'):
            raise RuntimeError('No setting things for now, thanks')

    def _send(self, obj):
        pickle.dump(obj, self.proc.stdin)
        self.proc.stdin.flush()

    def _recv(self):
        response_type, value = pickle.load(self.proc.stdout)

        if response_type == 'raise':
            raise value

        assert response_type == 'return'
        return value

    def __repr__(self):
        return '{}({})'.format(type(self).__name__,
                               self.calc_input)

    def __enter__(self):
        assert self.proc is None
        self.proc = self.mpi_command.execute()
        self._send(self.calc_input)
        return self

    def __exit__(self, *args):
        self._send('stop')
        self.proc.communicate()
        self.proc = None

    def _run_calculation(self, atoms, properties, system_changes):
        self._send('calculate')
        self._send((atoms, properties, system_changes))

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        # We send a pickle of self.atoms because this is a fresh copy
        # of the input, but without an unpicklable calculator:
        self._run_calculation(self.atoms.copy(), properties, system_changes)
        results = self._recv()
        self.results.update(results)

    def backend(self):
        return ParallelBackendInterface(self)


class MockMethod:
    def __init__(self, name, interface):
        self.name = name
        self.interface = interface

    def __call__(self, *args, **kwargs):
        ifc = self.interface
        ifc._send('callmethod')
        ifc._send([self.name, args, kwargs])
        return ifc._recv()


class ParallelBackendInterface:
    def __init__(self, interface):
        self.interface = interface

    def __getattr__(self, name):
        return MockMethod(name, self.interface)


run_modes = {'standard', 'mpi4py'}


def callmethod(calc, attrname, args, kwargs):
    method = getattr(calc, attrname)
    value = method(*args, **kwargs)
    return value


def calculate(calc, atoms, properties, system_changes):
    # Again we need formalization of the results/outputs, and
    # a way to programmatically access all available properties.
    # We do a wild hack for now:
    calc.results.clear()
    # If we don't clear(), the caching is broken!  For stress.
    # But not for forces.  What dark magic from the depths of the
    # underworld is at play here?
    calc.calculate(atoms=atoms, properties=properties,
                   system_changes=system_changes)
    results = calc.results
    return results


def bad_mode():
    return SystemExit(f'sys.argv[1] must be one of {run_modes}')


def main():
    try:
        run_mode = sys.argv[1]
    except IndexError:
        raise bad_mode()

    if run_mode not in run_modes:
        raise bad_mode()

    if run_mode == 'mpi4py':
        # We must import mpi4py before the rest of ASE, or world will not
        # be correctly initialized.
        import mpi4py  # noqa

    # We switch stdout so stray print statements won't interfere with outputs:
    binary_stdout = sys.stdout.buffer
    sys.stdout = sys.stderr

    from ase.parallel import world, broadcast

    def recv():
        if world.rank == 0:
            obj = pickle.load(sys.stdin.buffer)
        else:
            obj = None

        obj = broadcast(obj, 0, world)
        return obj

    def send(obj):
        if world.rank == 0:
            pickle.dump(obj, binary_stdout)
            binary_stdout.flush()

    pack = recv()
    calc = pack.unpack_calculator()

    while True:
        instruction = recv()
        if instruction == 'stop':
            return

        if instruction == 'callmethod':
            function = callmethod
        elif instruction == 'calculate':
            function = calculate
        else:
            raise RuntimeError(f'Bad instruction: {instruction}')

        instruction_data = recv()

        try:
            value = function(calc, *instruction_data)
        except Exception as ex:
            response_type = 'raise'
            value = ex
        else:
            response_type = 'return'

        send((response_type, value))


if __name__ == '__main__':
    main()
