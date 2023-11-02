import pytest

from ase.utils import (deprecated, devnull,
                       get_python_package_path_description, tokenize_version)


class MyWarning(UserWarning):
    pass


def _is_test_in_kwargs(*_, **kwargs) -> bool:
    return "test" in kwargs


@deprecated('hello', category=MyWarning)
def _add(a: int, b: int) -> int:
    return a + b


@deprecated('hello', category=MyWarning, condition=_is_test_in_kwargs)
def _subtract(a: int, b: int, *args, **kwargs) -> int:
    print(args, kwargs)
    return a - b


class TestDeprecatedDecorator:
    @staticmethod
    def test_should_raise_warning() -> None:
        with pytest.warns(MyWarning, match='hello'):
            assert _add(2, 2) == 4

    @staticmethod
    def test_should_raise_warning_when_test_in_kwargs() -> None:
        with pytest.warns(MyWarning, match='hello'):
            assert _subtract(2, 2, test=True) == 0

    @staticmethod
    def test_should_not_raise_warning_when_test_not_in_kwargs() -> None:
        assert _subtract(2, 2, not_test=True) == 0


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
