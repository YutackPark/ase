"""
Output support for X3D and X3DOM file types.
See http://www.web3d.org/x3d/specifications/
X3DOM outputs to html that display 3-d manipulatable atoms in
modern web browsers and jupyter notebooks.
"""

from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from ase.utils import writer
import xml.etree.ElementTree as ET
from xml.dom import minidom


@writer
def write_x3d(fd, atoms, format='X3D'):
    """Writes to html using X3DOM.

    Args:
        filename - str or file-like object, filename or output file object
        atoms - Atoms object to be rendered
        format - str, either 'X3DOM' for web-browser compatibility or 'X3D'
            to be readable by Blender. `None` to detect format based on file
            extension ('.html' -> 'X3DOM', '.x3d' -> 'X3D')"""
    X3D(atoms).write(fd, datatype=format)


@writer
def write_html(fd, atoms):
    """Writes to html using X3DOM.

    Args:
        filename - str or file-like object, filename or output file object
        atoms - Atoms object to be rendered"""
    write_x3d(fd, atoms, format='X3DOM')


class X3D:
    """Class to write either X3D (readable by open-source rendering
    programs such as Blender) or X3DOM html, readable by modern web
    browsers.
    """

    def __init__(self, atoms):
        self._atoms = atoms

    def write(self, fileobj, datatype):
        """Writes output to either an 'X3D' or an 'X3DOM' file, based on
        the extension. For X3D, filename should end in '.x3d'. For X3DOM,
        filename should end in '.html'.

        Args:
            datatype - str, output format. 'X3D' or 'X3DOM'.
        """

        if datatype == 'X3DOM':
            template = X3DOM_template
        elif datatype == 'X3D':
            template = X3D_template
        else:
            raise ValueError(f'datatype not supported: {datatype}')

        scene = x3d_atoms(self._atoms)
        document = template.format(scene=pretty_print(scene))
        print(document, file=fileobj)


def x3d_atoms(atoms):
    """Convert an atoms object into an x3d representation."""

    atom_spheres = [x3d_atom(atom) for atom in atoms]
    return element('scene', children=atom_spheres)


def element(name, child=None, children=None, **attributes) -> ET.Element:
    """Convenience function to make an XML element.

    If child is specified, it is appended to the element.
    If children is specified, they are appended to the element.
    You cannot specify both child and children."""

    # make sure we don't specify both child and children
    if child is not None:
        assert children is None, 'Cannot specify both child and children'
        children = [child]
    else:
        children = children or []

    element = ET.Element(name, **attributes)
    for child in children:
        element.append(child)
    return element


def x3d_atom(atom):
    """Represent an atom as an x3d, coloured sphere."""

    x, y, z = atom.position
    r, g, b = jmol_colors[atom.number]
    radius = covalent_radii[atom.number]

    material = element('material', diffuseColor=f'{r} {g} {b}')

    appearance = element('appearance', child=material)
    sphere = element('sphere', radius=f'{radius}')

    shape = element('shape', children=(appearance, sphere))

    return element('transform', translation=f'{x} {y} {z}', child=shape)


def pretty_print(element: ET.Element, indent: int = 2):
    """Pretty print an XML element."""

    byte_string = ET.tostring(element, 'utf-8')
    parsed = minidom.parseString(byte_string)
    prettied = parsed.toprettyxml(indent=' ' * indent)
    # remove first line - contains an extra, un-needed xml declaration
    lines = prettied.splitlines()[1:]
    return '\n'.join(lines)


X3DOM_template = """\
<html>
    <head>
        <title>ASE atomic visualization</title>
        <link rel="stylesheet" type="text/css" \
            href="https://www.x3dom.org/x3dom/release/x3dom.css"></link>
        <script type="text/javascript" \
            src="https://www.x3dom.org/x3dom/release/x3dom.js"></script>
    </head>
    <body>
        <X3D width="640px" height="480px">

<!--Inserting Generated X3D Scene-->
{scene}
<!--End of Inserted Scene-->

        </X3D>
    </body>
</html>
"""

X3D_template = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.2//EN" \
    "http://www.web3d.org/specifications/x3d-3.2.dtd">
<X3D profile="Interchange" version="3.2" \
    xmlns:xsd="http://www.w3.org/2001/XMLSchema-instance" \
    xsd:noNamespaceSchemaLocation=\
        "http://www.web3d.org/specifications/x3d-3.2.xsd">

<!--Inserting Generated X3D Scene-->
{scene}
<!--End of Inserted Scene-->

</X3D>
"""
