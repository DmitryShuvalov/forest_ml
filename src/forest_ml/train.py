from pathlib import Path
from re import X

import click
import pandas as pd
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
def train(
    csv_path: Path,
    target: str,
    random_state: int,
    test_split_ratio: int,
    drop_na: bool = True,
) -> None:
    X_train, X_val, y_train, y_val = get_splitted_dataset(
        csv_path, target, random_state, test_split_ratio, drop_na
    )
