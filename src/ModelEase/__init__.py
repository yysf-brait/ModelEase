n_jobs = 4
table = dict()

from .dataSet import *  # noqa
from .model import *  # noqa

import pandas as pd # noqa


def comparison() -> pd.DataFrame:
    return pd.DataFrame(table).T
