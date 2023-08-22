"""Inline viewer for jupyter notebook using X3D."""

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from IPython.display import HTML
from ase.io.x3d import write_x3d


def view_x3d(atoms, *args, **kwargs):
    """View atoms inline in a jupyter notbook. This command
    should only be used within a jupyter/ipython notebook.

    Args:
        atoms - ase.Atoms, atoms to be rendered"""

    notebook_style = {'width': '400px', 'height': '300px'}

    temp = StringIO()
    write_x3d(temp, atoms, format='X3DOM', style=notebook_style)
    data = temp.getvalue()
    temp.close()
    return HTML(data)
