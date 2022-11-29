"""unit tests for the turbomole reader module"""
# type: ignore
import pytest
from ase.calculators.turbomole.reader import parse_data_group


def test_parse_data_group():
    """test the parse_data_group() function in the turbomole reader module"""
    assert parse_data_group('', 'empty') is None
    assert parse_data_group('$name', 'name') == ''
    assert parse_data_group('$name val', 'name') == 'val'

    dgr_dct_s = {'start': '2.000'}
    dgr_dct_l = {'start': '2.000', 'step': '0.500', 'min': '1.000'}

    # with =
    dgr = '$scfdamp start=  2.000'
    assert parse_data_group(dgr, 'scfdamp') == dgr_dct_s
    dgr = '$scfdamp  start=  2.000  step =  0.500  min   = 1.000'
    assert parse_data_group(dgr, 'scfdamp') == dgr_dct_l

    # with spaces
    dgr = '$scfdamp  start  2.000'
    assert parse_data_group(dgr, 'scfdamp') == dgr_dct_s
    dgr = '$scfdamp  start 2.000  step 0.500  min  1.000'
    assert parse_data_group(dgr, 'scfdamp') == dgr_dct_l

    # with new lines and spaces
    dgr = '$scfdamp\n  start  2.000'
    assert parse_data_group(dgr, 'scfdamp') == dgr_dct_s
    dgr = '$scfdamp\n  start  2.000\n  step  0.500\n  min 1.000'
    assert parse_data_group(dgr, 'scfdamp') == dgr_dct_l

    # with new lines and mixture of = and spaces
    dgr = '$scfdamp\n  start = 2.000'
    assert parse_data_group(dgr, 'scfdamp') == dgr_dct_s
    dgr = '$scfdamp\n  start = 2.000  step 0.500 \n  min =  1.000'
    assert parse_data_group(dgr, 'scfdamp') == dgr_dct_l

    msg = r'data group does not start with \$empty'
    with pytest.raises(ValueError, match=msg):
        parse_data_group('$other', 'empty')
