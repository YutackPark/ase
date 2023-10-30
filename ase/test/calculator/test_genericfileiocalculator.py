"""Test suite for ase.calculators.GenericFileIOCalculator"""

from unittest.mock import patch, PropertyMock

import pytest

from ase.calculators.genericfileio import GenericFileIOCalculator


@pytest.mark.parametrize("calculator_kwargs, result_command",
                         [
                             ({"parallel": False}, ["dummy.x"]),
                             ({"parallel": False,  "parallel_info": {"-np": 4, "--oversubscribe": True}}, ["dummy.x"]),
                             ({"parallel": True,  "parallel_info": {"-np": 4, "--oversubscribe": False}},
                              ["mpirun", "-np", "4", "dummy.x"]),
                             ({"parallel": True}, ["mpirun", "dummy.x"]),
                             ({"parallel": True, "parallel_info": {"-np": 4, "--oversubscribe": True}},
                              ["mpirun", "-np", "4", "--oversubscribe", "dummy.x"]),
                         ])
def test_run_command(tmp_path, dummy_template, calculator_kwargs, result_command):
    """A test for the command creator from the config file"""

    from ase.config import Config
    mock_config = {
        "parallel": {
            "binary": "mpirun",
        },
        "dummy": {
            "exc": "dummy.x",
        }
    }

    with patch.object(Config,
                      'parser',
                      return_value=mock_config,
                      new_callable=PropertyMock):

        calc = GenericFileIOCalculator(template=dummy_template,
                                       profile=None,
                                       directory=tmp_path,
                                       **calculator_kwargs
                                       )
        assert calc.profile.get_command(inputfile="") == result_command

