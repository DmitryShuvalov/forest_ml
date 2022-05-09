from pathlib import Path
import warnings

import mlflow
import mlflow.sklearn

import click
from click import echo
from joblib import dump

import pandas as pd
import numpy as np
from sklearn import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score

from .helpers.data import get_splitted_dataset
from .helpers.pipeline import create_pipeline
from .helpers.search_parameters import get_parameters
from .eda import create_eda


@click.command()
@click.option(
    "-RS",
    "--random_state",
    default=42,
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
    "-US",
    "--use_scaler",
    default=True,
    type=bool,
    show_default=True,
    help="Dataset: Use StandartScaler",
)
@click.option(
    "-UP",
    "--use_pca",
    default=True,
    type=bool,
    show_default=True,
    help="Dataset: Use PCA",
)
@click.option(
    "-PNC",
    "--pca_n_components",
    default=2,
    type=click.IntRange(1),
    show_default=True,
    help="PCA: Parameter n_components",
)
@click.option(
    "-MN",
    "--model_name",
    default="RFC",
    type=click.Choice(["RFC", "DTC", "KNN"], case_sensitive=True),
    show_default=True,
    help='Model: Select model ""RFC"-RandomForesClassifier, "DTC"-DecisionTreeClassifier, "KNN"-KNeighborsClassifier',
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
    "-NN",
    "--n_neighbors",
    default=5,
    type=click.IntRange(0),
    show_default=True,
    help="Model: Parameter n_neighbors - WARNING: KNeighborsClassifier ONLY",
)
@click.option(
    "-W",
    "--weights",
    default="uniform",
    type=click.Choice(["uniform", "distance"], case_sensitive=True),
    show_default=True,
    help="Model: Parameter weights - WARNING: KNeighborsClassifier ONLY",
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
    "-UAHS",
    "--use_automatic_hyperparameter_search",
    default=False,
    type=bool,
    show_default=True,
    help="Automatic hyperparameter search using KFold cross-validation",
)
@click.option(
    "-NS",
    "--n_splits",
    default=3,
    type=click.IntRange(2),
    show_default=True,
    help="K-fold cross-validation: Parameter n_splits (number of splits)",
)
@click.option(
    "-NSO",
    "--n_splits_outer",
    default=5,
    type=click.IntRange(2),
    show_default=True,
    help="K-fold cross-validation: Parameter n_splits (number of splits) for cross_val_score when activated 'Automatic hyperparameter search'",
)
@click.option(
    "-NI",
    "--n_iter",
    default=10,
    type=click.IntRange(1),
    show_default=True,
    help="RandomizedSearchCV: parameter n_iter - Number of parameter settings that are sampled",
)
def train(
    csv_path: Path,
    target: str,
    random_state: int,
    test_split_ratio: int,
    model_name: str,
    drop_na: bool,
    use_scaler: bool,
    use_pca: bool,
    pca_n_components: int,
    n_estimators: int,
    criterion: str,
    splitter: str,
    max_depth: int,
    bootstrap: bool,
    max_features: int,
    output_file_path: Path,
    create_eda_report: bool,
    use_cross_validate: bool,
    use_automatic_hyperparameter_search: bool,
    n_splits: int,
    n_splits_outer: int,
    n_iter: int,
    save_model: bool,
    n_neighbors: int,
    weights: str,
) -> None:
    warnings.filterwarnings('ignore')
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

    if model_name == "RFC":
        run_name = "RandomForestClassifier"
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state,
            bootstrap=bootstrap,
            max_features=max_feat,
        )
    elif model_name == "DTC":
        run_name = "DecisionTreeClassifier"
        model = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            random_state=random_state,
            max_features=max_feat,
        )
    elif model_name == "KNN":
        run_name = "KNeighborsClassifier"
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, n_jobs=-1
        )
    else:
        raise Exception("Model doesn't exists", model_name)

    #Training with automatic hyperparameter search
    if use_automatic_hyperparameter_search:
        with mlflow.start_run(run_name="auto " + run_name):
            cv_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
            X = X_train#.append(X_val)
            y = y_train#.append(y_val)
            #part1 - search best hyperparameters
            cv_inner = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            space = get_parameters(model_name)
            search = RandomizedSearchCV(model, space, n_iter=n_iter, scoring="accuracy", n_jobs=-1, cv=cv_inner, refit=True)
            search.fit(X_train, y_train)
            #part2 - evaluate nested cross-validate for best_estimator
            scores = cross_val_score(search.best_estimator_, X_val, y_val, scoring='accuracy', cv=cv_outer, n_jobs=-1)
            print('best_score =', search.best_score_)
            print('best_params =', search.best_params_)
            print('NestedCV - Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
            
            mlflow.sklearn.log_model(search.best_estimator_, artifact_path="sklearn-model")
            mlflow.log_params(search.best_params_)
            y_pred = search.best_estimator_.predict(X_val)
            mlflow.log_metric("accuracy", accuracy_score(y_val, y_pred))
            mlflow.log_metric("precision", precision_score(y_val, y_pred, average="macro"))
            mlflow.log_metric("f1_score", f1_score(y_val, y_pred, average="macro"))
            mlflow.log_metric("nestedCV", np.mean(scores))
            
        return

    pipeline = create_pipeline(model, use_scaler, use_pca, pca_n_components)
    # MLFlow
    with mlflow.start_run(run_name=run_name):
        # Training without K-fold cross-validation
        pipeline.fit(X_train, y_train)
        acc_score_val = accuracy_score(y_val, pipeline.predict(X_val))
        pre_score_val = precision_score(y_val, pipeline.predict(X_val), average="macro")
        f1_score_val = f1_score(y_val, pipeline.predict(X_val), average="macro")
        mlflow.sklearn.log_model(pipeline, artifact_path="sklearn-model")
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("use_pca", use_pca)
        if use_pca:
            mlflow.log_param("pca_n_components", pca_n_components)
        if isinstance(
            model, RandomForestClassifier
        ):  # mlflow log for RandomForestClassifier
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("criterion", criterion)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("bootstrap", bootstrap)
            mlflow.log_param("max_features", max_feat)
            mlflow.log_param("random_state", random_state)
        elif isinstance(model, DecisionTreeClassifier):
            mlflow.sklearn.log_model(pipeline, artifact_path="sklearn-model")
            mlflow.log_param("criterion", criterion)
            mlflow.log_param("splitter", splitter)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("max_features", max_feat)
            mlflow.log_param("random_state", random_state)
        elif isinstance(model, KNeighborsClassifier):
            mlflow.log_param("n_neighbors", n_neighbors)
            mlflow.log_param("weights", weights)

        else:
            return
        mlflow.log_metric("accuracy", acc_score_val)
        mlflow.log_metric("precision", pre_score_val)
        mlflow.log_metric("f1_score", f1_score_val)
        echo("\nTraining without K-fold cross validation")
        echo(
            f"  Valid score: accuracy={acc_score_val:0.3f}, precision={pre_score_val:0.3f}, f1_score={f1_score_val:0.3f}\n"
        )

    # test = pd.read_csv('data/test.csv', index_col='Id')
    # test_pred=model.predict(test)
    # res=pd.DataFrame({'Id' : test.index.values, 'Cover_Type' : test_pred}).set_index('Id')
    # res.to_csv("data/test1.csv")

    # Save model to file
    if save_model:
        dump(pipeline, output_file_path)
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
            pipeline.fit(X_tr, y_tr)
            pred_values = pipeline.predict(X_test)
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

