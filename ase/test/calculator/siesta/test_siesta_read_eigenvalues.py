import ase.build
from ase.calculators.siesta import Siesta


def test_siesta_read_eigenvalues_soc(datadir, config_file):
    """ In this test, we read a stored siesta.EIG file."""
    calc = Siesta()
    assert calc.read_eigenvalues() == 1
    calc.directory = datadir / 'siesta'
    print(datadir)
    assert calc.read_eigenvalues() == 0
    assert calc.results['eigenvalues'].shape == (1, 1, 30)


def test_siesta_read_eigenvalues(siesta_factory):
    # Test real calculation which produces a gapped .EIG file
    atoms = ase.build.bulk('Si', cubic=True)
    calc = siesta_factory.calc(kpts=[2, 1, 1])
    atoms.calc = calc
    atoms.get_potential_energy()

    assert calc.results['eigenvalues'].shape[:2] == (1, 2)  # spins x bands
    assert calc.get_k_point_weights().shape == (2,)
    assert calc.get_ibz_k_points().shape == (2, 3)
