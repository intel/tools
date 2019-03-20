import pytest
import sys
from mock import patch

from test_utils.io import catch_stdout


@pytest.fixture
def mock_sys_args(patch):
    return patch("sys.argv")


def test_main(mock_sys_args):
    """Asserts passing in valid input results in return of 0"""
    mock_sys_args = ["--input", "/tmp"]
    import clean_control_input
    clean_control_input.main(["--flag", "arg"])


def test_main_bad_flags(mock_sys_args):
    """Asserts passing in bad input causes return of -1"""
    mock_sys_args = ["--input", "/notafile"]
    import clean_control_input
    with catch_stdout() as output:
        func_return = clean_control_input.main(["--flag", "arg"])
        output = output.getvalue()
    assert func_return == -1
    assert output == "Input graph file /notafile does not exist!"
