import unittest
import pytest
from unittest import mock

import numpy as np
from ase.calculators.vasp import Vasp
from ase.calculators.vasp.create_input import _args_without_comment
from ase.calculators.vasp.create_input import _to_vasp_bool, _from_vasp_bool

from ase.build import bulk

@pytest.fixture
def nacl():
    atoms = bulk('NaCl', crystalstructure='rocksalt', a=4.1,
                 cubic=True)
    return atoms



def test_vasp_potcar(nacl):
    def get_pp_symbols(ppp_list):
        mappings = {}
        for p in ppp_list:
            name = (p.split('/')[-2])
            element = name.split('_')[0] if '.' not in name else 'H'
            mappings[element] = name
        return mappings

    setups = {'recommended':
                  {'Na': 'Na_pv', 'Cl': 'Cl'},
              'GW':
                    {'Na': 'Na_sv_GW', 'Cl': 'Cl_GW'},
              'custom':
                    {'Na': 'Na', 'Cl': 'Cl_h'},
              }
    calc = Vasp(setups='recommended')
    calc.initialize(nacl)
    assert get_pp_symbols(calc.ppp_list) == setups['recommended']

    calc = Vasp(setups='GW')
    calc.initialize(nacl)
    assert get_pp_symbols(calc.ppp_list) == setups['GW']

    calc = Vasp(setups={'base': 'minimal', 'Cl': '_h'})
    calc.initialize(nacl)
    assert get_pp_symbols(calc.ppp_list) == setups['custom']
