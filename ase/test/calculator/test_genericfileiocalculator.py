"""Test suite for ase.calculators.GenericFileIOCalculator"""

import pytest

from ase.calculators.genericfileio import GenericFileIOCalculator
from ase.config import Config, cfg


@pytest.mark.parametrize(
    "calculator_kwargs, result_command",
    [
        ({"parallel": False}, ["dummy.x"]),
        (
            {
                "parallel": False,
                "parallel_info": {"-np": 4, "--oversubscribe": True}
            },
            ["dummy.x"],
        ),
        (
            {
                "parallel": True,
                "parallel_info": {"-np": 4, "--oversubscribe": False}
            },
            ["mpirun", "-np", "4", "dummy.x"],
        ),
        ({"parallel": True}, ["mpirun", "dummy.x"]),
        (
            {
                "parallel": True,
                "parallel_info": {"-np": 4, "--oversubscribe": True}
            },
            ["mpirun", "-np", "4", "--oversubscribe", "dummy.x"],
        ),
        (
            {"parallel": True, "parallel_info": {"nprocs": 4}},
            ["mpirun", "-np", "4", "dummy.x"],
        ),
    ],
)
def test_run_command(
        tmp_path, dummy_template, calculator_kwargs, result_command,
        monkeypatch,
):
    """A test for the command creator from the config file"""

    mock_config = Config()
    mock_config.parser.update({
        "parallel": {"binary": "mpirun", "nprocs_kwarg_trans": "-np"},
        "dummy": {
            "exc": "dummy.x",
        },
    })

    monkeypatch.setattr(cfg, 'parser', mock_config.parser)
    calc = GenericFileIOCalculator(
        template=dummy_template,
        profile=None,
        directory=tmp_path,
        **calculator_kwargs
    )
    assert calc.profile.get_command(inputfile="") == result_command
