import numpy as np
from ase import Atoms

def run_optimization(relax, fmax: float = 0.05, smax: float = 0.00005,
                     smask=np.array([1, 1, 1, 1, 1, 1]),
                     logfile='relax.log') -> Atoms:
    """
    Use the BGFS ASE optimizer + ExpCellFilter to perform various types of
    structure optimizations.
    Can provide custom force and stress convergence criteria to the
    optimizer when needed.

    :param atoms: Atoms object with an ASE calculator object attached to the
        atoms.
    :param fmax: Force cut off.
    :param smax: Stress tensor cut off.
    :param smask: 1x6 np.array used to select which diagonal components of the
        stress tensor to allow to relax.
        0=ignored, 1=checked.
        By specifying various smask values you can perform different
        types of structure/geometric optimizations:
            Example: fixed cell: np.array([0, 0, 0, 0, 0, 0]),
            full volume relax: np.array([1, 1, 1, 1, 1, 1]),
            2D (xy plane): np.array([1, 1, 0, 0, 0, 1]),
            1D periodic: np.array([0, 0, 1, 0, 0, 0])
    :param identifier: The restart, trajectory, and log file name.
    :param comm: Parallel communicator to properly write files.
    :return: {'atoms': atoms.copy(), 'traj_path': Path('traj')}
    """
    atoms = relax.atoms.atoms
    with relax as opt, open(logfile, 'a') as log_fd:
        # we hardcode the restart file because doing both mag and non-mag
        # runs, we expect them to use the same hessian. We do this for the
        # logfile because of errors in the forces/stresses written logfile
        print('{:12s} {:17s} {:18s} {:16s} {:6s}'.format(
            'Step', 'Time', 'Energy', 'fmax', 'smax'), flush=True)
        print('{:12s} {:17s} {:18s} {:16s} {:6s}'.format(
            'Step', 'Time', 'Energy', 'fmax', 'smax'), file=log_fd, flush=True)

        # set force cut off convergence; apply custom criteria inside irun
        for i, _ in enumerate(opt.irun(fmax=0)):
            # params to write out to log: e, f, & s
            f = atoms.get_forces()
            fmaxnow = np.linalg.norm(f, axis=1).max()**0.5
            s = atoms.get_stress() * smask
            smaxnow = abs(s).max()
            e = atoms.get_potential_energy(force_consistent=True)

            # time stamp
            from datetime import datetime
            now = str(datetime.now()).rsplit('.', 1)[0]  # rm frac secs

            # loop break condition
            done = fmaxnow <= fmax and smaxnow <= smax

            print(f'{i:4d} {now} {e:16.6f} {fmaxnow:16.6f} {smaxnow:16.6f}',
                  flush=True)
            print(f'{i:4d} {now} {e:16.6f} {fmaxnow:16.6f} {smaxnow:16.6f}',
                  file=log_fd, flush=True)
            # check if calculation is done
            if done:
                opt.call_observers()
                break
    return atoms

