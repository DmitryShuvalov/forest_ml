from ast import Str
from turtle import shape
import pandas as pd
import numpy as np


def create_fake_dataset(target: str) -> pd.DataFrame():
    shape0 = 100
    shape1_binary = 10
    shape1_int = 50
    df_binary = pd.DataFrame(
        np.random.randint(0, 2, size=(shape0, shape1_binary)),
        columns=range(1, 1 + shape1_binary),
    )
    df_int = pd.DataFrame(
        np.random.randint(0, 3000, size=(shape0, shape1_int)),
        columns=range(1 + shape1_binary, 1 + shape1_binary + shape1_int),
    )
    df = pd.concat([df_binary, df_int], axis=1)
    df["Id"] = range(shape0)
    df[target] = np.random.randint(1, 10, size=shape0)
    df.set_index("Id")
    return df
