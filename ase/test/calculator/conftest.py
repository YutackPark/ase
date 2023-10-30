from os import PathLike
from typing import Mapping, Any

import pytest

from ase.calculators.genericfileio import CalculatorTemplate


@pytest.fixture(autouse=True)
def _calculator_tests_always_use_testdir(testdir):
    pass


class DummyTemplate(CalculatorTemplate):

    def __init__(self):
        super().__init__(
            name="dummy",
            implemented_properties=()
        )

    def write_input(self, directory, atoms, parameters, properties):
        pass

    def load_profile(self, cfg):
        pass

    def execute(self, directory, profile):
        pass

    def read_results(self, directory: PathLike) -> Mapping[str, Any]:
        pass


@pytest.fixture
def dummy_template():
    return DummyTemplate()
