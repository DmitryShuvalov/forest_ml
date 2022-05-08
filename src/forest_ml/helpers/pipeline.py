from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def create_pipeline(model, use_scaler=True, use_selector=True, n_components=2):
    steps=[]
    if use_scaler:
        steps.append(('scaler', StandardScaler()))
    if use_selector:
        steps.append(('selector', PCA(n_components=n_components)))
    steps.append(('model', model))
    return Pipeline(steps)