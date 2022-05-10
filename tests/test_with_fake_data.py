from forest_ml import __version__
from forest_ml.train import train
import pytest
from click.testing import CliRunner
from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score
from .helpers.create_fake_dataset import create_fake_dataset


""" Testing with fake data and file system isolation"""


def test_version():
    assert __version__ == "0.1.0"


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_model_name(runner: CliRunner) -> None:
    """It fails when model_name not in ['KNN', 'RFC', 'DTC']."""
    with runner.isolated_filesystem():
        target = "target"
        create_fake_dataset(target=target).to_csv("temp.csv")
        result = runner.invoke(
            train,
            [
                "--csv_path",
                "temp.csv",
                "--target",
                target,
                "--model_name",
                "Fail",
                "--save_model",
                False,
            ],
        )
        assert result.exit_code == 2
        assert "Invalid value for '-MN' / '--model_name'" in result.output


def test_success_for_valid_model_name(runner: CliRunner) -> None:
    """It fails when model_name not in ['KNN', 'RFC', 'DTC']."""
    with runner.isolated_filesystem():
        target = "target"
        create_fake_dataset(target=target).to_csv("temp.csv")
        result = runner.invoke(
            train,
            [
                "--csv_path",
                "temp.csv",
                "--target",
                target,
                "--model_name",
                "KNN",
                "--save_model",
                False,
            ],
        )
        assert result.exit_code == 0


def test_success_for_valid_saved_model(runner: CliRunner) -> None:
    """It passes when accuracy saved_model.predict>0.5."""
    with runner.isolated_filesystem():
        target = "target"
        saved_model_name = "saved_model.joblib"
        create_fake_dataset(target=target).to_csv("temp.csv")
        result = runner.invoke(
            train,
            [
                "--csv_path",
                "temp.csv",
                "--target",
                target,
                "--model_name",
                "RFC",
                "--save_model",
                True,
                "--output_file_path",
                saved_model_name,
            ],
        )
        saved_model = load(saved_model_name)
        df = pd.read_csv("temp.csv", index_col="Id")
        X = df.drop(columns=[target])
        y = df[target]
        res = accuracy_score(y, saved_model.predict(X))
        assert res >= 0.5
        assert result.exit_code == 0
