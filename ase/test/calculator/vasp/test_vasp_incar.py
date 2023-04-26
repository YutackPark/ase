import pytest
from ase.atoms import Atoms

calc = pytest.mark.calculator



@calc('vasp')
def test_vasp_incar(factory):
    """Test that INCAR is written correctly."""
    # Have each of every type of key in settings
    settings = {'xc': 'PBE',  # special str key
                'encut': 400,  # Float key
                'ediff': 1e-6,  # Exp key
                'ibrion': 2,  # Int key
                'prec': 'Low',  # str key
                'lattice_constraints': [False, True, False],  # list_bool key
                'iband': [1, 2, 3],  # list_int key
                'lhfcalc': True,  # bool key
                'lreal': True,  # special key
                'magmom': [.5, 1.5],  # list_float key
                'ldau_luj': {'H': {'L': 2, 'U': 4.0, 'J': 0.0},}  # dict key
                }
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
    
