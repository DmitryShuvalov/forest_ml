from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def create_pipeline(model, use_scaler=True, use_pca=True, n_components=2):
    steps = []
    if use_scaler:
        steps.append(("scaler", MinMaxScaler()))
    if use_pca:
        steps.append(("selector", PCA(n_components=n_components)))
    steps.append(("model", model))
    return Pipeline(steps)
