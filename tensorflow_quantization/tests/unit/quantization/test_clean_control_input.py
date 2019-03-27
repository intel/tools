import pytest
from mock import MagicMock

from quantization.clean_control_input import main
from test_utils.io import catch_stdout


@pytest.fixture()
def mock_gfile(patch):
    return patch("gfile")


@pytest.fixture(autouse=True)
def mock_flags(patch):
    return patch("FLAGS")


@pytest.fixture
def mock_graph_pb2(patch):
    return patch("graph_pb2")


def test_main(mock_gfile, mock_graph_pb2):
    """Asserts passing in valid input results in return of 0"""
    mock_gfile.Exists.return_value = True
    mock_graph_pb2.GraphDef.return_value.node = [MagicMock(input=['a']), MagicMock(input=['^', 'b'])]
    assert main(["--flag", "arg"]) == 0


@pytest.mark.parametrize('flag_input', ['/notafile', '%00', '碁'])
def test_main_bad_flags(flag_input, mock_gfile, mock_flags):
    """Asserts passing in bad input causes return of -1"""
    mock_gfile.Exists.return_value = False
    mock_flags.input = flag_input
    with catch_stdout() as output:
        func_return = main(["--flag", "arg"])
        output = output.getvalue()
    assert func_return == -1
    assert output == "Input graph file '{}' does not exist!\n".format(flag_input)
