from forest_ml import __version__
from forest_ml.train import train
import pytest
from click.testing import CliRunner

# def test_version():
#     assert __version__ == '0.1.0'
    

@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_model_name(
    runner: CliRunner
) -> None:
    """It fails when model_name not in ['KNN', 'RFC', 'DTC']."""
    result = runner.invoke(
        train,
        [
            "--model_name",
            "FAIL",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '-MN' / '--model_name'" in result.output

def test_ok_for_valid_model_name(
    runner: CliRunner
) -> None:
    """It fails when model_name not in ['KNN', 'RFC', 'DTC']."""
    result = runner.invoke(
        train,
        [
            "--model_name",
            "KNN",
        ],
    )
    assert result.exit_code == 0
    #assert "Invalid value for '-MN' / '--model_name'" in result.output
