import pytest
from ase.build import bulk


def test_x3d():
    pytest.importorskip('IPython')
    from ase.visualize import x3d
    from IPython.display import HTML
    atoms = bulk('Cu', cubic=True)
    my_obj = x3d.view_x3d(atoms)
    assert isinstance(my_obj, HTML)
