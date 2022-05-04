from pathlib import Path
from tkinter.tix import X_REGION

import click
from click import echo
from joblib import dump
from numpy import average

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import KFold

from .helpers.data import get_splitted_dataset
from .eda import create_eda


@click.command()
@click.option(
    "-RS",
    "--random_state",
    default=None,
    type=int,
    show_default=True,
    help = 'General: Random_state',
)
@click.option(
    "-CER",
    "--create_eda_report",
    default=False,
    type=bool,
    show_default=True,
    help = 'EDA report: Create and save to data/eda.html file/ For more parameters use "poetry run eda"',
)
@click.option(
    "-SM",
    "--save_model",
    default=True,
    type=bool,
    show_default=True,
    help = 'Model: Save trained model to file (path in parameter --output_file_path)',
)
@click.option(
    "-OFP",
    "--output_file_path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, path_type=Path, writable=True),
    show_default=True,
    help = 'Model: Path for saveng trained model (switch in parameter --save_model)',
)
@click.option(
    "-CP",
    "--csv_path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
    help = 'Dataset: Path to source train csv file',
)
@click.option(
    "--target",
    default="Cover_Type",
    type=str,
    show_default=True,
    help = 'Dataset: Name of target column',
)
@click.option(
    "-TSR",
    "--test_split_ratio",
    default=0.2,
    type=float,
    show_default=True,
    help = 'Dataset: Test size ratio for splitting dataset, when train whithout K-fold cross-validation',
)
@click.option(
    "-DN",
    "--drop_na",
    default=True,
    type=bool,
    show_default=True,
    help = 'Dataset: Drop NA values',
)
@click.option(
    "-NE",
    "--n_estimators",
    default=100,
    type=click.IntRange(
        1,
    ),
    show_default=True,
    help = 'RandomForestClassifier: Parameter n_estimators',

)
@click.option(
    "-C",
    "--criterion",
    default="gini",
    type=click.Choice(["gini", "entropy"], case_sensitive=True),
    show_default=True,
    help = 'RandomForestClassifier: Parameter criterion "gini" or "entropy"',
)
@click.option(
    "-NJ",
    "--n_jobs",
    default=-1,
    type=click.IntRange(-1),
    show_default=True,
    help = 'RandomForestClassifier: Parameter n_jobs (-1 if max available)',
)
@click.option(
    "-NS",
    "--n_splits",
    default=5,
    type=click.IntRange(2),
    show_default=True,
    help = 'K-folda cross-validation: Parameter n_splits (number of splits)',
)

def train(
    csv_path: Path,
    target: str,
    random_state: int,
    test_split_ratio: int,
    drop_na: bool,
    n_estimators: int,
    criterion: str,
    n_jobs: int,
    output_file_path: Path,
    create_eda_report: bool,
    n_splits: int,
    save_model: bool
) -> None:
    if create_eda_report:
        eda_report_path = create_eda(from_csv=csv_path)
        echo(f"EDA report is saved to {eda_report_path}\n")

    X_train, X_val, y_train, y_val = get_splitted_dataset(
        csv_path, target, random_state, test_split_ratio, drop_na
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        n_jobs=n_jobs,
    )

    # Training without K-fold cross-validation
    model.fit(X_train, y_train)
    acc_score_train = accuracy_score(y_train, model.predict(X_train))
    acc_score_val = accuracy_score(y_val, model.predict(X_val))
    pre_score_train = precision_score(y_train, model.predict(X_train), average="micro")
    pre_score_val = precision_score(y_val, model.predict(X_val), average="micro")
    f1_score_train = f1_score(y_train, model.predict(X_train), average="micro")
    f1_score_val = f1_score(y_val, model.predict(X_val), average="micro")

    echo("\nTraining without K-fold cross validation")
    echo(
        f"  Train score: accuracy={acc_score_train:0.3f}, precision={pre_score_train:0.3f}, f1_score={f1_score_train:0.3f}"
    )
    echo(
        f"  Valid score: accuracy={acc_score_val:0.3f}, precision={pre_score_val:0.3f}, f1_score={f1_score_val:0.3f}\n"
    )

    # Training with K-fold cross-validation
    X = X_train.append(X_val)
    y = y_train.append(y_val)

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    kf.split(X)
    scores = pd.DataFrame({"accuracy": [], "precision": [], "f1_score": []})
    for train_index, test_index in kf.split(X):
        X_tr, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_tr, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_tr, y_tr)
        pred_values = model.predict(X_test)
        scores = scores.append(
            {
                "accuracy": accuracy_score(pred_values, y_test),
                "precision": precision_score(pred_values, y_test, average="micro"),
                "f1_score": f1_score(pred_values, y_test, average="micro"),
            },
            ignore_index=True,
        )

    echo("\nTraining without K-fold cross validation")
    echo(scores)

    # Save model to file
    if save_model:
        dump(model, output_file_path)
        echo(f"Model is saved to {output_file_path}.")
