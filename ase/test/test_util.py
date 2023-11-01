import pytest

from ase.utils import (deprecated, devnull,
                       get_python_package_path_description, tokenize_version)


def test_deprecated_decorator():

    class MyWarning(UserWarning):
        pass

    @deprecated('hello', MyWarning)
    def add(a, b):
        return a + b

    with pytest.warns(MyWarning, match='hello'):
        assert add(2, 2) == 4


def test_deprecated_devnull():
    with pytest.warns(DeprecationWarning):
        devnull.tell()


@pytest.mark.parametrize('v1, v2', [
    ('1', '2'),
    ('a', 'b'),
    ('9.0', '10.0'),
    ('3.8.0', '3.8.1'),
    ('3a', '3b'),
    ('3', '3a'),
])
def test_tokenize_version_lessthan(v1, v2):
    v1 = tokenize_version(v1)
    v2 = tokenize_version(v2)
    assert v1 < v2


def test_tokenize_version_equal():
    version = '3.8x.xx'
    assert tokenize_version(version) == tokenize_version(version)


class DummyIterator:
    def __iter__(self):
        yield from ["test", "bla"]


class Dummy:
    @property
    def __path__(self):
        return DummyIterator()


def test_get_python_package_path_description():
    assert isinstance(get_python_package_path_description(Dummy()), str)
    # test object not containing __path__
    assert isinstance(get_python_package_path_description(object()), str)
