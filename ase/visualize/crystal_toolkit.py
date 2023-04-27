from ase import Atoms
from contextlib import contextmanager


class CrystalToolKitDisplaySetting:
    """Storage space for the display settings"""

    def __init__(self):
        self.scene_kwargs = {}
        self.legend_kwargs = dict(color_scheme="Jmol", radius_scheme="uniform")
        self.with_bonds = True
        self.bond_nn_class = None

    def apply(self, **kwargs):
        """
        Apply settings from called arguments

        Example
        
            >>> settings.apply(with_bonds=False, bond_nn_class=CrystalNN)

        """
        for key, value in kwargs.items():
            attr = getattr(self, key)
            if isinstance(value, dict):
                attr.update(value)
            else:
                setattr(self, key, value)

    def to_dict(self):
        """Return a dictionary of the settings"""

        return {
            'scene_kwargs': self.scene_kwargs,
            'legend_kwargs': self.legend_kwargs,
            'with_bonds': self.with_bonds,
            'bond_nn_class': self.bond_nn_class,
        }



DISPLAY_SETTINGS = CrystalToolKitDisplaySetting()

@contextmanager
def display_option(**kwargs):
    """
    Context manage for applying display settings temporarily within a with block.

    Example

        >>> with display_option(with_bonds=False):
                obj = view(atoms, viewer="crystal_toolkit")
        >>> obj
    """

    backup = DISPLAY_SETTINGS.to_dict()
    DISPLAY_SETTINGS.apply(**kwargs)
    yield
    DISPLAY_SETTINGS.apply(**backup)


class CrystalToolKitDisplay:
    """
    Display using Crystal-Toolkit
    """

    def __init__(
        self,
        atoms: Atoms,
        with_bonds=None,
        bond_nn_class=None,
        scene_kwargs=None,
        legend_kwargs=None,
    ):
        """
        Instantiate a CrystalToolKitDisplay object

        Args:
            atoms: The Atoms object to be viewed.
            with_bond: Include the bonding in the display.
            bond_nn_class: 
                NearestNeighbour class to be used for constructing the connectivity.
                Defaults to pymatgen.analysis.local_env import MinimumDistanceNN.
            scene_kwargs: 
                A dictionary containing the key words passed to get_structure_scene
                or get_structure_graph_scene.
        """

        from pymatgen.analysis.local_env import MinimumDistanceNN
        from pymatgen.io.ase import AseAtomsAdaptor

        bond_nn_class = (
            bond_nn_class
            if bond_nn_class is not None
            else DISPLAY_SETTINGS.bond_nn_class
        )

        if bond_nn_class is None:
            self.bond_nn_class = MinimumDistanceNN

        self.scene_kwargs = (
            scene_kwargs
            if scene_kwargs is not None
            else DISPLAY_SETTINGS.scene_kwargs
        )
        self.legend_kwargs = (
            legend_kwargs
            if legend_kwargs is not None
            else DISPLAY_SETTINGS.legend_kwargs
        )

        self.atoms = atoms
        self.ps = AseAtomsAdaptor.get_structure(atoms)
        self.with_bonds = (
            with_bonds
            if with_bonds is not None
            else DISPLAY_SETTINGS.with_bonds
        )

    def build_scene(self):
        """Build the scene for using display"""
        from crystal_toolkit.renderables import StructureGraph
        from crystal_toolkit.core.legend import Legend
        from crystal_toolkit.renderables.structuregraph import (
            get_structure_graph_scene,
        )
        from crystal_toolkit.renderables.structure import get_structure_scene

        if self.with_bonds:
            # Patch the get_scene method
            graph = StructureGraph.with_local_env_strategy(
                self.ps, self.bond_nn_class()
            )
            graph.get_scene = lambda: get_structure_graph_scene(
                graph,
                **self.scene_kwargs,
                legend=Legend(self.ps, **self.legend_kwargs)
            )

            return graph
        # Patch the get_scene method
        self.ps.get_scene = lambda: get_structure_scene(
            self.ps,
            **self.scene_kwargs,
            legend=Legend(self.ps, **self.legend_kwargs)
        )

        return self.ps


def view_crystal_toolkit(atoms, **kwargs):
    """View with crystal tookit"""
    return CrystalToolKitDisplay(atoms, **kwargs).build_scene()
