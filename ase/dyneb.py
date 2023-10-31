"""Temporary file while we deprecate this locaation."""

from ase.mep import DyNEB as RealDyNEB
from ase.utils import deprecated


class DyNEB(RealDyNEB):
    @deprecated('Please import DyNEB from ase.mep, not ase.dyneb.')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
