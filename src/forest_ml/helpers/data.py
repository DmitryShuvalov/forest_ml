import pandas as pd
from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
from typing import Tuple
from click import echo


def get_splitted_dataset(
    csv_path: Path,
    target: str,
    random_state: int,
    test_split_ratio: float,
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path, index_col="Id")
    echo(f"Dataset shape: {dataset.shape}.")
    # if use_ohe:
    #     dataset = OneHotEncoder().fit()
    # There may be a mechanism for filling in missing data. Now we know that such data yet do not exist.
    if drop_na:
        dataset.dropna(inplace=True)
    X = dataset.drop(target, axis=1)
    y = dataset[target]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_split_ratio, random_state=random_state
    )
    return X_train, X_val, y_train, y_val
