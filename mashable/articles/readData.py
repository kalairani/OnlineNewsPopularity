import pandas as pd
import pkg_resources


def read(file):
    filepath=pkg_resources.resource_filename(__name__, file)
    return pd.read_csv(filepath)

