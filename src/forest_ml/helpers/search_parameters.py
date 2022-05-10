def get_parameters(model_name: str) -> dict():
    if model_name == "RFC":
        return {
            "n_estimators": [100, 200, 500, 1000],
            "criterion": ["gini", "entropy"],
            "max_depth": [10, 20, 100],
            "max_features": ["auto", "sqrt", "log2", 5, 10, 50],
        }
    elif model_name == "DTC":
        return {
            "splitter": ["best", "random"],
            "criterion": ["gini", "entropy"],
            "max_depth": [10, 20, 100],
            "max_features": ["auto", "sqrt", "log2", 5, 10, 50],
        }
    elif model_name == "KNN":
        return {
            "n_neighbors": [2, 3, 5, 7, 10, 15],
            "weights": ["uniform", "distance"],
        }
    else:
        return None
