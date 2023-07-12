import ase.parallel as parallel
from ase.visualize.viewers import VIEWERS


def view(atoms, data=None, viewer='ase', repeat=None, block=False, **kwargs):
    if parallel.world.size > 1:
        return

    vwr = VIEWERS[viewer.lower()]
    handle = vwr.view(atoms, data=data, repeat=repeat, **kwargs)

    if block and hasattr(handle, 'wait'):
        status = handle.wait()
        if status != 0:
            raise RuntimeError(f'Viewer "{vwr.name}" failed with status '
                               '{status}')

    return handle
