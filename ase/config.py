from collections.abc import Mapping
import os


class ConfigVars(Mapping):
    def __init__(self):
        self._dct = os.environ

    def __iter__(self):
        return iter(self._dct)

    def __getitem__(self, item):
        return self._dct[item]

    def __len__(self):
        return len(self._dct)


cfg = ConfigVars()
