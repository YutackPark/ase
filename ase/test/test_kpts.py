def test_kpts():
    import numpy as np
    from ase.dft.kpoints import bandpath
    print(bandpath('GX,GX', np.eye(3), 6))
