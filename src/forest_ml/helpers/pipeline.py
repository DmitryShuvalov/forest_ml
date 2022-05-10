from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from typing import Any


def create_pipeline(
    model: Any, use_scaler: bool = True, use_pca: bool = True, n_components: int = 2
) -> Pipeline:
    steps = []
    if use_scaler:
        steps.append(("scaler", MinMaxScaler()))
    if use_pca:
        steps.append(("selector", PCA(n_components=n_components)))
    steps.append(("model", model))
    return Pipeline(steps)
