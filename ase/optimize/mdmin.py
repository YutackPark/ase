from typing import IO, Optional, Union

import numpy as np

from ase import Atoms
from ase.optimize.optimize import Optimizer


class MDMin(Optimizer):
    # default parameters
    defaults = {**Optimizer.defaults, 'dt': 0.2}

    def __init__(
        self,
        atoms: Atoms,
        restart: Optional[str] = None,
        logfile: Union[IO, str] = '-',
        trajectory: Optional[str] = None,
        dt: Optional[float] = None,
        master: Optional[bool] = None,
    ):
        """Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: string
            Text file used to write summary information.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.
        """
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)

        if dt is None:
            self.dt = self.defaults['dt']
        else:
            self.dt = dt

    def initialize(self):
        self.v = None

    def read(self):
        self.v, self.dt = self.load()

    def step(self, forces=None):
        atoms = self.atoms

        if forces is None:
            forces = atoms.get_forces()

        if self.v is None:
            self.v = np.zeros((len(atoms), 3))
        else:
            self.v += 0.5 * self.dt * forces
            # Correct velocities:
            vf = np.vdot(self.v, forces)
            if vf < 0.0:
                self.v[:] = 0.0
            else:
                self.v[:] = forces * vf / np.vdot(forces, forces)

        self.v += 0.5 * self.dt * forces
        pos = atoms.get_positions()
        atoms.set_positions(pos + self.dt * self.v)
        self.dump((self.v, self.dt))
