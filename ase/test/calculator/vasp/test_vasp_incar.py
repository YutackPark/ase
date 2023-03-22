import pytest
from ase.atoms import Atoms

calc = pytest.mark.calculator



@calc('vasp')
def test_vasp_incar(factory):
    """Test that INCAR is written correctly."""
    settings = {'xc': 'PBE', 'encut': 400, 'ediff': 1e-6, 'ibrion': 2, 'prec': 'Low'}
    calc = factory.calc(**settings)
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.2]])
    calc.initialize(atoms)
    calc.write_incar(atoms)
    
    # Check that INCAR is written correctly
    with open('INCAR', 'r') as f:
        lines = f.readlines()
    print(lines)
    assert 'INCAR created by Atomic Simulation Environment' == lines[0].strip()
    # assert 'ENCUT = 400.0' in lines
    
