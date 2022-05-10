from forest_ml import __version__
from forest_ml.train import train
import pytest
from click.testing import CliRunner

''' Testing with fake data fith file system isolation'''
def test_version():
     assert __version__ == '0.1.0'

@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


