from pathlib import Path

import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .helpers.data import get_splitted_dataset


@click.command()
@click.option(
    "-CP",
    "--csv_path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "--target",
    default="Cover_Type",
    type=str,
    show_default=True,
)
@click.option(
    "-RS",
    "--random_state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "-TSR",
    "--test_split_ratio",
    default=0.2,
    type=float,
    show_default=True,
)
@click.option(
    "-DN",
    "--drop_na",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "-NE",
    "--n_estimators",
    default=100,
    type=click.IntRange(1, ),
    show_default=True,
)
@click.option(
    "-C",
    "--criterion",
    default='gini',
    type=click.Choice(['gini', 'entropy'], case_sensitive=True),
    show_default=True,
)
@click.option(
    "-NJ",
    "--n_jobs",
    default=-1,
    type=click.IntRange(-1, ),
    show_default=True,
)
def train(
    csv_path: Path,
    target: str,
    random_state: int,
    test_split_ratio: int,
    drop_na: bool,
    n_estimators:int,
    criterion:str,
    n_jobs:int

) -> None:
    X_train, X_val, y_train, y_val = get_splitted_dataset(
        csv_path, target, random_state, test_split_ratio, drop_na
    )

    model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, n_jobs=n_jobs,)

 
