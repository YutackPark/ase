# File solely for gently deprecating this BaseSiesta class.
from ase.calculators.siesta.siesta import Siesta
from ase.utils import deprecated


class BaseSiesta(Siesta):
    @deprecated(
        "The BaseSiesta calculator class will no longer be supported. "
        "Use `ase.calculators.siesta.Siesta` instead.",
        category=FutureWarning,
    )
    def __init__(self, *args, **kwargs):
        """
        .. deprecated:: 3.18.2
            The ``BaseSiesta`` calculator class will no longer be supported.
            Use :class:`~ase.calculators.siesta.Siesta` instead.
        """
        Siesta.__init__(self, *args, **kwargs)
