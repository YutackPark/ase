import unittest
import pytest
from unittest import mock

import numpy as np
from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.calculators.vasp.create_input import _args_without_comment
from ase.calculators.vasp.create_input import _to_vasp_bool, _from_vasp_bool

from ase.build import bulk

calc = pytest.mark.calculator
@pytest.fixture
def nacl():
    atoms = bulk('NaCl', crystalstructure='rocksalt', a=4.1,
                 cubic=True) * (3, 3, 3)
    return atoms


@pytest.fixture
def vaspinput_factory(nacl):
    """Factory for GenerateVaspInput class, which mocks the generation of
    pseudopotentials."""
    def _vaspinput_factory(atoms=None, **kwargs) -> GenerateVaspInput:
        if atoms is None:
            atoms = nacl
        mocker = mock.Mock()
        inputs = GenerateVaspInput()
        inputs.set(**kwargs)
        inputs._build_pp_list = mocker(return_value=None)  # type: ignore
        inputs.initialize(atoms)
        return inputs

    return _vaspinput_factory

@calc('vasp')
def test_vasp_potcar(nacl):
    vaspinputs = GenerateVaspInput(setups='recommended')
    assert vaspinputs.ppp_list == ['Na_pv', 'Cl']