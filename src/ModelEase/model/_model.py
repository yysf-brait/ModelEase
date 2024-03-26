import pandas as pd

model_list = {}


def comparison() -> pd.DataFrame:
    return pd.DataFrame(model_list).T
