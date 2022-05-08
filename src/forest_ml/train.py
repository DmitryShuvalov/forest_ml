from pathlib import Path

import mlflow
import mlflow.sklearn

import click
from click import echo
from joblib import dump

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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
    help="General: Random_state",
)
@click.option(
    "-CER",
    "--create_eda_report",
    default=False,
    type=bool,
    show_default=True,
    help='EDA report: Create and save to data/eda.html file/ For more parameters use "poetry run eda"',
)
@click.option(
    "-SM",
    "--save_model",
    default=True,
    type=bool,
    show_default=True,
    help="Model: Save trained model to file (path in parameter --output_file_path)",
)
@click.option(
    "-OFP",
    "--output_file_path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, path_type=Path, writable=True),
    show_default=True,
    help="Model: Path for saveng trained model (switch in parameter --save_model)",
)
@click.option(
    "-CP",
    "--csv_path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
    help="Dataset: Path to source train csv file",
)
@click.option(
    "--target",
    default="Cover_Type",
    type=str,
    show_default=True,
    help="Dataset: Name of target column",
)
@click.option(
    "-TSR",
    "--test_split_ratio",
    default=0.2,
    type=float,
    show_default=True,
    help="Dataset: Test size ratio for splitting dataset, when train whithout K-fold cross-validation",
)
@click.option(
    "-DN",
    "--drop_na",
    default=True,
    type=bool,
    show_default=True,
    help="Dataset: Drop NA values",
)
@click.option(
    "-MN",
    "--model_name",
    default="RFC",
    type=click.Choice(["RFC", "DTC"], case_sensitive=True),
    show_default=True,
    help='Model: Select model ""RFC"-RandomForesClassifier or "DTC"-DecisionTreeClassifier',
)
@click.option(
    "-NE",
    "--n_estimators",
    default=100,
    type=click.IntRange(
        1,
    ),
    show_default=True,
    help="Model: Parameter n_estimators - WARNING: RandomForesClassifier ONLY",
)
@click.option(
    "-C",
    "--criterion",
    default="gini",
    type=click.Choice(["gini", "entropy"], case_sensitive=True),
    show_default=True,
    help='Model: Parameter criterion "gini" or "entropy"',
)
@click.option(
    "-SP",
    "--splitter",
    default="best",
    type=click.Choice(["best", "random"], case_sensitive=True),
    show_default=True,
    help='Model: Parameter splitter "best" or "random" - WARNING: DecisionTreeClassifier',
)
@click.option(
    "-MD",
    "--max_depth",
    default=None,
    type=click.IntRange(1),
    show_default=True,
    help="Model: Parameter max_depth",
)
@click.option(
    "-MF",
    "--max_features",
    default=0,
    type=click.IntRange(-2),
    show_default=True,
    help="Model: Parameter max_features (0=auto, -1=sqrt, -2=log2, other - just int values)",
)
@click.option(
    "-BS",
    "--bootstrap",
    default=True,
    type=bool,
    show_default=True,
    help="Model: Parameter bootstrap - WARNING: RandomForesClassifier ONLY",
)
@click.option(
    "-UCV",
    "--use_cross_validate",
    default=True,
    type=bool,
    show_default=True,
    help="K-fold cross-validation: Execute k-fold cross-validation",
)
@click.option(
    "-NS",
    "--n_splits",
    default=5,
    type=click.IntRange(2),
    show_default=True,
    help="K-fold cross-validation: Parameter n_splits (number of splits)",
)
def train(
    csv_path: Path,
    target: str,
    random_state: int,
    test_split_ratio: int,
    model_name: str,
    drop_na: bool,
    n_estimators: int,
    criterion: str,
    splitter: str, 
    max_depth: int,
    bootstrap: bool,
    max_features: int,
    output_file_path: Path,
    create_eda_report: bool,
    use_cross_validate: bool,
    n_splits: int,
    save_model: bool,
) -> None:
    if create_eda_report:
        eda_report_path = create_eda(from_csv=csv_path)
        echo(f"EDA report is saved to {eda_report_path}\n")

    X_train, X_val, y_train, y_val = get_splitted_dataset(
        csv_path, target, random_state, test_split_ratio, drop_na
    )

    if max_features == -2:
        max_feat = "log2"
    elif max_features == -1:
        max_feat = "sqrt"
    elif max_features == 0:
        max_feat = "auto"
    else:
        max_feat = max_features
    
    if model_name =="RFC":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state,
            bootstrap=bootstrap,
            max_features=max_feat,
        )
    elif model_name =="DTC":
        model = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            random_state=random_state,
            max_features=max_feat,
        )
    else:
        raise Exception('Not such model', model_name)

        # MLFlow
    run_name = "RandomForestClassifier" if model_name=="RFC" else "DecisionTreeClassifier"
    with mlflow.start_run(run_name=run_name):
        # Training without K-fold cross-validation
        model.fit(X_train, y_train)
        acc_score_val = accuracy_score(y_val, model.predict(X_val))
        pre_score_val = precision_score(y_val, model.predict(X_val), average="macro")
        f1_score_val = f1_score(y_val, model.predict(X_val), average="macro")
        if isinstance(model, RandomForestClassifier):   #mlflow log for RandomForestClassifier
            mlflow.sklearn.log_model(model, artifact_path="sklearn-model")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("criterion", criterion)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("bootstrap", bootstrap)
            mlflow.log_param("max_features", max_feat)
            mlflow.log_param("random_state", random_state)    
        elif isinstance(model, DecisionTreeClassifier):
            mlflow.sklearn.log_model(model, artifact_path="sklearn-model")
            mlflow.log_param("criterion", criterion)
            mlflow.log_param('splitter', splitter)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("max_features", max_feat)
            mlflow.log_param("random_state", random_state)    
        else:
            return
        mlflow.log_metric("accuracy", acc_score_val)
        mlflow.log_metric("precision", pre_score_val)
        mlflow.log_metric("f1_score", f1_score_val)
        echo("\nTraining without K-fold cross validation")
        echo(
            f"  Valid score: accuracy={acc_score_val:0.3f}, precision={pre_score_val:0.3f}, f1_score={f1_score_val:0.3f}\n"
        )

    # Save model to file
    if save_model:
        dump(model, output_file_path)
        echo(f"Model is saved to {output_file_path}.")

    # Training with K-fold cross-validation
    if use_cross_validate:
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
                    "precision": precision_score(pred_values, y_test, average="macro"),
                    "f1_score": f1_score(pred_values, y_test, average="macro"),
                },
                ignore_index=True,
            )
        echo("Training with K-fold cross-validate")
        echo(scores)
